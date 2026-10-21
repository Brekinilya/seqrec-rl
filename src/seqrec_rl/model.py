from pathlib import Path

import numpy as np
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from .data import N_SPECIAL, PAD


def build_model(n_tokens: int, max_len: int = 256, d_model: int = 256, n_layer: int = 4,
                n_head: int = 4, dropout: float = 0.1) -> GPT2LMHeadModel:
    cfg = GPT2Config(vocab_size=n_tokens, n_positions=max_len, n_embd=d_model, n_layer=n_layer,
                     n_head=n_head, resid_pdrop=dropout, embd_pdrop=dropout, attn_pdrop=dropout,
                     bos_token_id=None, eos_token_id=None, attn_implementation="sdpa")
    return GPT2LMHeadModel(cfg)


def save_model(model: GPT2LMHeadModel, path: str | Path, **extra) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": model.config.to_dict(), "state": model.state_dict(), **extra}, path)


def load_model(path: str | Path, device: str = "cuda") -> tuple[GPT2LMHeadModel, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = GPT2Config.from_dict(ckpt["config"])
    model = GPT2LMHeadModel(cfg)
    model.load_state_dict(ckpt["state"])
    return model.to(device), ckpt


def hidden_states(model: GPT2LMHeadModel, input_ids: torch.Tensor) -> torch.Tensor:
    # Sequences are right-padded and attention is causal, so real positions never see padding.
    return model.transformer(input_ids=input_ids).last_hidden_state


class LinearCrossEntropy(torch.autograd.Function):
    """sum_i CE(h_i @ w.T, y_i) without materialising the N x V logit matrix.

    Gradients are computed chunk by chunk during forward (softmax - onehot is
    known there), so backward only rescales them and nothing is recomputed.
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.bfloat16)
    def forward(ctx, h, w, y, chunk):
        grad_h = torch.empty_like(h)
        grad_w = torch.zeros(w.shape, dtype=torch.float32, device=w.device)
        loss = torch.zeros((), dtype=torch.float32, device=h.device)
        for i in range(0, len(y), chunk):
            hc, yc = h[i:i + chunk], y[i:i + chunk]
            logits = (hc @ w.t()).float()
            lse = torch.logsumexp(logits, dim=-1)
            loss += (lse - logits.gather(1, yc[:, None]).squeeze(1)).sum()
            p = logits.sub_(lse[:, None]).exp_()
            p[torch.arange(len(yc), device=p.device), yc] -= 1
            p = p.to(h.dtype)
            grad_h[i:i + chunk] = p @ w
            grad_w += (p.t() @ hc).float()
        ctx.save_for_backward(grad_h, grad_w)
        return loss

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_out):
        grad_h, grad_w = ctx.saved_tensors
        return grad_h * grad_out, grad_w * grad_out, None, None


def next_item_loss(model: GPT2LMHeadModel, input_ids: torch.Tensor, labels: torch.Tensor,
                   chunk: int = 2048) -> tuple[torch.Tensor, int]:
    """Full-softmax next-item cross-entropy (mean over targets); PAD and OOV targets are skipped."""
    h = hidden_states(model, input_ids)
    keep = labels >= N_SPECIAL
    h, y = h[keep], labels[keep]
    n = len(y)
    loss = LinearCrossEntropy.apply(h, model.lm_head.weight, y, chunk)
    return loss / max(n, 1), n


def pad_right(seqs: list[np.ndarray], max_len: int) -> tuple[np.ndarray, np.ndarray]:
    lengths = np.array([min(len(s), max_len) for s in seqs])
    out = np.full((len(seqs), lengths.max()), PAD, dtype=np.int64)
    for i, s in enumerate(seqs):
        s = s[-max_len:]
        out[i, :len(s)] = s
    return out, lengths


@torch.no_grad()
def last_logits(model: GPT2LMHeadModel, contexts: list[np.ndarray], device: str = "cuda") -> torch.Tensor:
    """Next-item logits after each context; special tokens get -inf."""
    ids, lengths = pad_right(contexts, model.config.n_positions)
    ids = torch.from_numpy(ids).to(device)
    lengths = torch.from_numpy(lengths).to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = hidden_states(model, ids)
        h = h[torch.arange(len(lengths), device=device), lengths - 1]
        logits = (h @ model.lm_head.weight.t()).float()
    logits[:, :N_SPECIAL] = float("-inf")
    return logits


class GPTRecommender:
    """Scores the whole vocabulary from the last `n_positions` tokens of each user's history."""

    def __init__(self, model: GPT2LMHeadModel, device: str = "cuda"):
        self.model, self.device = model, device

    def scores(self, es, idx: np.ndarray) -> torch.Tensor:
        self.model.eval()
        return last_logits(self.model, es.contexts(idx, self.model.config.n_positions), self.device)


class WindowSampler:
    """Cuts every user's sequence into windows of seq_len + 1 tokens with a random phase per epoch,
    so each target position is used exactly once per epoch."""

    def __init__(self, offsets: np.ndarray, tokens: np.ndarray, seq_len: int, seed: int = 0):
        self.offsets, self.tokens, self.seq_len = offsets, tokens, seq_len
        self.rng = np.random.default_rng(seed)

    def windows(self) -> np.ndarray:
        L = self.seq_len
        out = []
        for s, e in zip(self.offsets[:-1], self.offsets[1:]):
            if e - s < 2:
                continue
            starts = np.arange(s - self.rng.integers(0, L), e - 1, L)
            out.append(np.stack([np.maximum(starts, s), np.minimum(starts + L + 1, e)], 1))
        w = np.concatenate(out)
        return w[self.rng.permutation(len(w))]

    def batches(self, batch_size: int):
        w = self.windows()
        for i in range(0, len(w), batch_size):
            chunk = w[i:i + batch_size]
            batch = np.full((len(chunk), self.seq_len + 1), PAD, dtype=np.int64)
            for j, (s, e) in enumerate(chunk):
                batch[j, :e - s] = self.tokens[s:e]
            yield batch

    def n_batches(self, batch_size: int) -> int:
        n = sum(int(np.ceil((e - s - 1) / self.seq_len)) + 1 for s, e in zip(self.offsets[:-1], self.offsets[1:]))
        return n // batch_size
