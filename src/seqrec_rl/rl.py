"""KL-regularised reward optimisation of the next-item policy.

Single-step bandit: the state is the user's history, the action the next track,
the reward a reward-model prediction for that track. The objective is

    J(pi) = E_s [ E_{a ~ pi(.|s)} r(s, a) - beta * KL(pi(.|s) || pi_ref(.|s)) ]

whose maximiser is known in closed form, pi*(a|s) ~ pi_ref(a|s) exp(r(s, a) / beta).
Because the action space is a finite vocabulary and the reward model scores all
of it at once, both the objective and its gradient can be computed exactly.
"""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .data import N_SPECIAL
from .model import hidden_states, pad_right
from .reward import EventArrays, RewardModel

NEG = float("-inf")


@dataclass(frozen=True)
class Reward:
    heads: tuple[str, ...]
    fn: callable  # (probs: dict[str, (B, V)], hist: (B, V) bool, track_len: (V,)) -> (B, V)


REWARDS = {
    "ratio": Reward(("ratio",), lambda p, hist, tl: p["ratio"]),
    "like": Reward(("like",), lambda p, hist, tl: p["like"]),
    "like_plus_ratio": Reward(("like", "ratio"), lambda p, hist, tl: p["like"] + 0.1 * p["ratio"]),
    "seconds": Reward(("ratio",), lambda p, hist, tl: p["ratio"] * tl),
    "cont": Reward(("cont",), lambda p, hist, tl: p["cont"]),
    "relisten": Reward(("relisten",), lambda p, hist, tl: p["relisten"]),
    # a new track the user will come back to on their own within a week
    "discovery": Reward(("relisten",), lambda p, hist, tl: p["relisten"] * (~hist)),
}


@torch.no_grad()
def rm_rewards(model: RewardModel, h: torch.Tensor, hist: torch.Tensor, track_len: torch.Tensor,
               names) -> dict[str, torch.Tensor]:
    heads = sorted({hd for n in names for hd in REWARDS[n].heads})
    probs = {k: torch.sigmoid(v) for k, v in model.score_all(h, hist, heads).items()}
    return {n: REWARDS[n].fn(probs, hist, track_len) for n in names}


@torch.no_grad()
def encode_windows(model: RewardModel, ev: EventArrays, windows: np.ndarray) -> torch.Tensor:
    """Reward-model state after the last event of each window [w0, w1)."""
    max_len = model.config.n_positions
    windows = np.stack([np.maximum(windows[:, 0], windows[:, 1] - max_len), windows[:, 1]], 1)
    dev = next(model.parameters()).device
    b = {k: v.to(dev) for k, v in ev.gather(windows, max_len).items() if k in EventArrays.INPUT}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = model.encode(b["token"].long(), b["ratio_bucket"].long(), b["flags"].long(), b["gap_bucket"].long())
    last = torch.as_tensor(windows[:, 1] - windows[:, 0] - 1, device=dev)
    return h[torch.arange(len(windows), device=dev), last].float()


def policy_logits(model, contexts: list[np.ndarray], device: str = "cuda") -> torch.Tensor:
    """Next-item logits after each context (with gradients if enabled); special tokens get -inf."""
    ids, lengths = pad_right(contexts, model.config.n_positions)
    ids = torch.from_numpy(ids).to(device)
    lengths = torch.from_numpy(lengths).to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = hidden_states(model, ids)
        h = h[torch.arange(len(lengths), device=device), lengths - 1]
        logits = (h @ model.lm_head.weight.t()).float()
    mask = torch.zeros(logits.shape[1], dtype=torch.bool, device=device)
    mask[:N_SPECIAL] = True
    return logits.masked_fill(mask, NEG)


def pi_star_logp(ref_logp: torch.Tensor, r: torch.Tensor, beta: float,
                 candidates: torch.Tensor | None = None) -> torch.Tensor:
    """log pi* for reward r already divided by its scale; candidates restricts the support."""
    z = ref_logp + r / beta
    if candidates is not None:
        z = z.masked_fill(~candidates, NEG)
    return torch.log_softmax(z, dim=1)


def kl(logp: torch.Tensor, ref_logp: torch.Tensor) -> torch.Tensor:
    p = logp.exp()
    diff = torch.where(p > 0, logp - ref_logp, torch.zeros_like(logp))
    return (p * diff).sum(1)


def expectation(logp: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    p = logp.exp()
    return (p * torch.nan_to_num(r, neginf=0.0)).sum(1)


def exact_loss(logits: torch.Tensor, ref_logp: torch.Tensor, r: torch.Tensor, beta: float):
    """-E_pi[r] + beta KL(pi || pi_ref), averaged over states; exact over the vocabulary."""
    logp = torch.log_softmax(logits, dim=1)
    er, k = expectation(logp, r), kl(logp, ref_logp)
    return (-er + beta * k).mean(), er.detach(), k.detach()


def sampled_loss(logits: torch.Tensor, ref_logp: torch.Tensor, r: torch.Tensor, beta: float, group: int):
    """REINFORCE with a leave-one-out group baseline (unbiased for the same objective) + exact KL."""
    logp = torch.log_softmax(logits, dim=1)
    with torch.no_grad():
        a = torch.multinomial(logp.exp(), group, replacement=True)
        ra = r.gather(1, a)
        baseline = (ra.sum(1, keepdim=True) - ra) / (group - 1)
        adv = ra - baseline
    pg = -(adv * logp.gather(1, a)).mean(1)
    k = kl(logp, ref_logp)
    return (pg + beta * k).mean(), expectation(logp, r).detach(), k.detach()


def cutoff_states(rm_events_path, uids: np.ndarray, cutoff: int, keep: int = 520):
    """Reward-model event arrays with each user's last `keep` events before cutoff, and uid -> row."""
    import polars as pl

    from .reward import make_event_arrays
    df = (pl.read_parquet(rm_events_path)
          .filter(pl.col("uid").is_in(uids.tolist()) & (pl.col("timestamp") < cutoff))
          .sort("uid", "timestamp")
          .with_columns(pos=pl.int_range(pl.len(), dtype=pl.Int64).over("uid"),
                        n=pl.len().over("uid").cast(pl.Int64))
          .filter(pl.col("pos") >= pl.col("n") - keep).drop("pos", "n"))
    arrays = make_event_arrays(df, cutoff)
    return arrays, {int(u): i for i, u in enumerate(arrays.uids)}


class TrainStates:
    """Random (user, time) states from the training events; policy and reward model see the same history."""

    def __init__(self, ev: EventArrays, max_len: int, min_history: int = 16, seed: int = 0):
        self.ev, self.max_len = ev, max_len
        self.rng = np.random.default_rng(seed)
        lens = np.diff(ev.offsets)
        self.users = np.flatnonzero(lens > min_history)
        self.min_history = min_history

    def sample(self, n: int):
        users = self.rng.choice(self.users, n)
        s, e = self.ev.offsets[users], self.ev.offsets[users + 1]
        # position of the last event in the state, uniform over the user's events
        t = s + self.min_history + (self.rng.random(n) * (e - s - self.min_history)).astype(np.int64)
        return self.batch(s, t + 1)

    def batch(self, starts: np.ndarray, ends: np.ndarray):
        tok = self.ev.cols["token"]
        contexts = [tok[max(a, b - self.max_len):b].astype(np.int64) for a, b in zip(starts, ends)]
        history = [np.unique(tok[a:b][tok[a:b] >= N_SPECIAL]) for a, b in zip(starts, ends)]
        windows = np.stack([starts, ends], 1)
        return contexts, history, windows


@torch.no_grad()
def reward_scales(states: TrainStates, ref_model, rm: RewardModel, track_len: torch.Tensor, names,
                  n_states: int = 512, batch: int = 32) -> dict[str, float]:
    """Typical within-state spread of each reward under pi_ref: mean_s sqrt(Var_{pi_ref}[r | s])."""
    from .reward import history_mask
    acc = {n: [] for n in names}
    n_tokens = ref_model.config.vocab_size
    for _ in range(n_states // batch):
        contexts, history, windows = states.sample(batch)
        ref_logp = torch.log_softmax(policy_logits(ref_model, contexts), 1)
        hist = history_mask(history, n_tokens)
        rs = rm_rewards(rm, encode_windows(rm, states.ev, windows), hist, track_len, names)
        p = ref_logp.exp()
        for n, r in rs.items():
            m = (p * r).sum(1, keepdim=True)
            acc[n].append(((p * (r - m) ** 2).sum(1)).sqrt().cpu())
    return {n: float(torch.cat(v).mean()) for n, v in acc.items()}
