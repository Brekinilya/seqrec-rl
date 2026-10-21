"""Reward model: predicts the user's response to a track given the listening history.

History positions carry the track and the observed response to it (played-ratio
bucket, like/dislike, organic flag, time gap). The head is factorised so that a
state can be scored against the whole vocabulary with a few matrix products:

    logit_k(s, a) = <u_k(s), e_a> + c_k(s) + b_k(a) + len_k(a) [+ organic_k(s)]

with separate (u, c) for tracks the user has and has not listened to before.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model

from .data import N_SPECIAL, PAD
from .reward_data import HEADS, N_GAP, N_LEN, N_RATIO, UNKNOWN_GAP, UNKNOWN_RATIO

ORGANIC_BIT = 4
BASE_RATES = {"ratio": 0.65, "listen_plus": 0.63, "like": 0.0043, "dislike": 0.0012, "cont": 0.45, "relisten": 0.28}


class RewardModel(nn.Module):
    def __init__(self, n_tokens: int, len_bucket: np.ndarray, max_len: int = 512, d_model: int = 256,
                 n_layer: int = 4, n_head: int = 4, dropout: float = 0.1):
        super().__init__()
        cfg = GPT2Config(vocab_size=n_tokens, n_positions=max_len, n_embd=d_model, n_layer=n_layer,
                         n_head=n_head, resid_pdrop=dropout, embd_pdrop=dropout, attn_pdrop=dropout,
                         bos_token_id=None, eos_token_id=None, attn_implementation="sdpa")
        self.config = cfg
        self.gpt = GPT2Model(cfg)
        self.d, self.H = d_model, len(HEADS)
        self.ratio_emb = nn.Embedding(N_RATIO, d_model)
        self.flag_emb = nn.Embedding(8, d_model)
        self.gap_emb = nn.Embedding(N_GAP, d_model)
        self.u = nn.Linear(d_model, 2 * self.H * d_model)
        self.c = nn.Linear(d_model, 2 * self.H)
        self.org = nn.Linear(d_model, self.H)
        self.item_bias = nn.Embedding(n_tokens, self.H)
        self.len_bias = nn.Embedding(N_LEN, self.H)
        self.register_buffer("len_bucket", torch.as_tensor(len_bucket, dtype=torch.long), persistent=False)
        for emb in (self.ratio_emb, self.flag_emb, self.gap_emb, self.item_bias, self.len_bias):
            nn.init.zeros_(emb.weight)
        nn.init.normal_(self.u.weight, std=0.02 / d_model ** 0.5)
        nn.init.zeros_(self.u.bias)
        rates = torch.tensor([BASE_RATES[k] for k in HEADS])
        with torch.no_grad():
            self.c.weight.zero_()
            self.c.bias.copy_(torch.logit(rates).repeat(2))

    def init_from_policy(self, state: dict) -> None:
        """Copy the transformer and item embeddings of a trained GPT2LMHeadModel."""
        sub = {k[len("transformer."):]: v for k, v in state.items() if k.startswith("transformer.")}
        self.gpt.load_state_dict(sub)

    def encode(self, tokens, ratio_b, flags, gap_b) -> torch.Tensor:
        x = self.gpt.wte(tokens) + self.ratio_emb(ratio_b) + self.flag_emb(flags) + self.gap_emb(gap_b)
        return self.gpt(inputs_embeds=x).last_hidden_state

    def logits(self, h, item, rep, org) -> torch.Tensor:
        """(N, d) states, (N,) candidates -> (N, H) logits."""
        n = h.shape[0]
        ar = torch.arange(n, device=h.device)
        sel = rep.long()
        u = self.u(h).view(n, 2, self.H, self.d)[ar, sel]
        c = self.c(h).view(n, 2, self.H)[ar, sel]
        e = self.gpt.wte(item)
        out = (u * e[:, None, :]).sum(-1) + c + self.item_bias(item) + self.len_bias(self.len_bucket[item])
        return out + org.float()[:, None] * self.org(h)

    def score_all(self, h, rep_mask, heads=HEADS) -> dict[str, torch.Tensor]:
        """(B, d) states, (B, V) bool mask of tracks already in the history -> {head: (B, V) logits},
        candidates scored as recommendations (organic flag off)."""
        b = h.shape[0]
        idx = [HEADS.index(k) for k in heads]
        U = self.u(h).view(b, 2, self.H, self.d)
        C = self.c(h).view(b, 2, self.H)
        E = self.gpt.wte.weight
        bias = self.item_bias.weight + self.len_bias.weight[self.len_bucket]
        out = {}
        for k, j in zip(heads, idx):
            new = U[:, 0, j] @ E.t() + C[:, 0, j, None]
            rep = U[:, 1, j] @ E.t() + C[:, 1, j, None]
            out[k] = torch.where(rep_mask, rep, new) + bias[None, :, j]
        return out


def save_rm(model: RewardModel, path: str | Path, **extra) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": model.config.to_dict(), "state": model.state_dict(), **extra}, path)


def load_rm(path: str | Path, len_bucket: np.ndarray, device: str = "cuda") -> tuple[RewardModel, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    c = ckpt["config"]
    model = RewardModel(c["vocab_size"], len_bucket, c["n_positions"], c["n_embd"], c["n_layer"], c["n_head"],
                        c["resid_pdrop"])
    model.load_state_dict(ckpt["state"])
    return model.to(device).eval(), ckpt


class EventArrays:
    """Column arrays of reward-model events, concatenated per user (sorted by time)."""

    INPUT = ("token", "ratio_bucket", "flags", "gap_bucket")
    TARGET = ("is_repeat", "is_organic", "ratio", "lp", "liked", "disliked", "cont", "relisten", "relisten_valid")
    PAD_VALUES = {"token": PAD, "ratio_bucket": UNKNOWN_RATIO, "gap_bucket": UNKNOWN_GAP}

    def __init__(self, cols: dict[str, np.ndarray], offsets: np.ndarray, uids: np.ndarray):
        self.cols, self.offsets, self.uids = cols, offsets, uids

    def gather(self, windows: np.ndarray, width: int) -> dict[str, torch.Tensor]:
        out = {}
        for name, arr in self.cols.items():
            buf = np.full((len(windows), width), self.PAD_VALUES.get(name, 0), dtype=arr.dtype)
            for j, (s, e) in enumerate(windows):
                buf[j, :e - s] = arr[s:e]
            out[name] = torch.from_numpy(buf)
        return out


def batch_outputs(model: RewardModel, batch: dict, feedback_dropout: float = 0.0):
    """Teacher-forced: the state after event i is scored against event i + 1.

    Returns logits (N, H), labels (N, H), weights (N, H) and the flat index of
    each target inside the (B, L) target grid.
    """
    dev = next(model.parameters()).device
    b = {k: v.to(dev, non_blocking=True) for k, v in batch.items()}
    tok, rb, fl, gp = b["token"][:, :-1], b["ratio_bucket"][:, :-1].long(), b["flags"][:, :-1].long(), \
        b["gap_bucket"][:, :-1].long()
    if feedback_dropout > 0:
        # teach the model to cope with positions whose response is unknown (generated items)
        drop = torch.rand(rb.shape, device=dev) < feedback_dropout
        rb = torch.where(drop, UNKNOWN_RATIO, rb)
        fl = torch.where(drop, fl & ORGANIC_BIT, fl)
        gp = torch.where(torch.rand(gp.shape, device=dev) < feedback_dropout, UNKNOWN_GAP, gp)
    h = model.encode(tok.long(), rb, fl, gp)

    tgt = b["token"][:, 1:].long()
    m = tgt >= N_SPECIAL
    flat = torch.nonzero(m.reshape(-1)).squeeze(1)
    logits = model.logits(h[m], tgt[m], b["is_repeat"][:, 1:][m], b["is_organic"][:, 1:][m])
    t = {k: v[:, 1:][m] for k, v in b.items()}
    labels = torch.stack([
        t["ratio"].float().clamp(0, 100) / 100, t["lp"].float(), t["liked"].float(), t["disliked"].float(),
        t["cont"].float(), t["relisten"].float(),
    ], 1)
    weights = torch.ones_like(labels)
    weights[:, HEADS.index("relisten")] = t["relisten_valid"].float()
    return logits, labels, weights, flat


def rm_loss(model: RewardModel, batch: dict, feedback_dropout: float) -> tuple[torch.Tensor, torch.Tensor]:
    logits, labels, weights, _ = batch_outputs(model, batch, feedback_dropout)
    bce = F.binary_cross_entropy_with_logits(logits.float(), labels, reduction="none") * weights
    per_head = bce.sum(0) / weights.sum(0).clamp(min=1)
    return per_head.sum(), per_head.detach()


@torch.no_grad()
def encode_last(model: RewardModel, ev: EventArrays, users: np.ndarray, max_len: int) -> torch.Tensor:
    """State after the last event of each user (users are row numbers in ev)."""
    s, e = ev.offsets[users], ev.offsets[users + 1]
    windows = np.stack([np.maximum(s, e - max_len), e], 1)
    b = {k: v.to(next(model.parameters()).device) for k, v in ev.gather(windows, max_len).items()
         if k in EventArrays.INPUT}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = model.encode(b["token"].long(), b["ratio_bucket"].long(), b["flags"].long(), b["gap_bucket"].long())
    last = torch.as_tensor(windows[:, 1] - windows[:, 0] - 1, device=h.device)
    return h[torch.arange(len(users), device=h.device), last].float()


def history_mask(history: list[np.ndarray], n_tokens: int, device: str = "cuda") -> torch.Tensor:
    rows = np.concatenate([np.full(len(x), i) for i, x in enumerate(history)])
    cols = np.concatenate(history)
    mask = torch.zeros(len(history), n_tokens, dtype=torch.bool, device=device)
    mask[torch.from_numpy(rows).to(device), torch.from_numpy(cols).to(device)] = True
    return mask


def make_event_arrays(df, label_cutoff: int) -> EventArrays:
    """df: rows of rm_events.parquet sorted by (uid, timestamp)."""
    from .reward_data import relisten_labels

    lens = df.group_by("uid", maintain_order=True).len()
    offsets = np.zeros(lens.height + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lens["len"].to_numpy())
    relisten, valid = relisten_labels(df["timestamp"].to_numpy().astype(np.float64),
                                      df["next_lp_ts"].to_numpy().astype(np.float64), label_cutoff)
    cols = {
        "token": df["token"].to_numpy().astype(np.int32),
        "ratio_bucket": df["ratio_bucket"].to_numpy(),
        "flags": df["flags"].to_numpy(),
        "gap_bucket": df["gap_bucket"].to_numpy(),
        "is_repeat": df["is_repeat"].to_numpy(),
        "is_organic": df["is_organic"].to_numpy(),
        "ratio": df["played_ratio_pct"].to_numpy().astype(np.int16),
        "lp": df["lp"].to_numpy(),
        "liked": df["liked"].to_numpy(),
        "disliked": df["disliked"].to_numpy(),
        "cont": df["cont"].to_numpy(),
        "relisten": relisten,
        "relisten_valid": valid,
    }
    return EventArrays(cols, offsets, lens["uid"].to_numpy())


@torch.no_grad()
def predict_targets(model: RewardModel, ev: EventArrays, targets: list[tuple[int, int]], max_len: int,
                    batch_size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Teacher-forced probabilities for events [t0, t1) of every user (global indices).

    Targets are processed in chunks of at most max_len // 2, so each one sees at
    least half a context of real history when it exists.
    """
    model.eval()
    step = max_len // 2
    jobs = []  # (window start, window end, first kept target)
    for (t0, t1), s in zip(targets, ev.offsets[:-1]):
        for a in range(t0, t1, step):
            e = min(a + step, t1)
            jobs.append((max(s, e - max_len - 1), e, a))
    out = {}
    for i in range(0, len(jobs), batch_size):
        chunk = jobs[i:i + batch_size]
        windows = np.array([(w0, w1) for w0, w1, _ in chunk])
        batch = ev.gather(windows, max_len + 1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _, _, flat = batch_outputs(model, batch)
        probs = torch.sigmoid(logits.float()).cpu().numpy()
        flat = flat.cpu().numpy()
        row, col = flat // max_len, flat % max_len
        for (w0, w1, a), r_idx in zip(chunk, range(len(chunk))):
            sel = np.flatnonzero(row == r_idx)
            ev_idx = w0 + 1 + col[sel]
            keep = ev_idx >= a
            for g, p in zip(ev_idx[keep], probs[sel[keep]]):
                out[g] = p
    idx = np.array(sorted(out))
    return idx, np.stack([out[g] for g in idx]) if len(idx) else np.zeros((0, len(HEADS)))
