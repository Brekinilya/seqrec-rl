import numpy as np
import polars as pl
import torch

from .data import DAY, LISTEN_PLUS_PCT, N_SPECIAL, TemporalSplit
from .evaluation import EvalSet


class MostPop:
    """Most listened tracks over the last `days` before the evaluation cutoff."""

    def __init__(self, lis: pl.DataFrame, split: TemporalSplit, part: str, n_tokens: int, days: float,
                 device: str = "cuda"):
        cutoff = split.train_end if part == "val" else split.val_end
        recent = lis.filter(pl.col("timestamp").is_between(cutoff - int(days * DAY), cutoff, closed="left")
                            & (pl.col("token") >= N_SPECIAL))
        counts = np.bincount(recent["token"].to_numpy(), minlength=n_tokens).astype(np.float32)
        # a tiny all-time prior breaks ties among tracks unseen in the window
        prior = np.arange(n_tokens, 0, -1, dtype=np.float32) / n_tokens * 1e-3
        self.base = torch.from_numpy(counts + prior).to(device)

    def scores(self, es: EvalSet, idx: np.ndarray) -> torch.Tensor:
        return self.base.expand(len(idx), -1).clone()


class PersonalRepeat:
    """Re-recommends the user's own tracks by exponentially decayed listen counts,
    then falls back to MostPop. gamma is the decay per listen."""

    def __init__(self, pop: MostPop, gamma: float, listen_plus_only: bool):
        self.pop, self.gamma, self.listen_plus_only = pop, gamma, listen_plus_only

    def scores(self, es: EvalSet, idx: np.ndarray) -> torch.Tensor:
        base = self.pop.scores(es, idx)
        s = base / (base.max() + 1)
        rows, cols, vals = [], [], []
        for j, i in enumerate(idx):
            t, r = es.hist_tokens[i], es.hist_ratio[i]
            w = self.gamma ** np.arange(len(t) - 1, -1, -1, dtype=np.float64)
            keep = t >= N_SPECIAL
            if self.listen_plus_only:
                keep &= r >= LISTEN_PLUS_PCT
            rows.append(np.full(keep.sum(), j))
            cols.append(t[keep])
            vals.append(w[keep])
        rows = torch.from_numpy(np.concatenate(rows)).to(s.device)
        cols = torch.from_numpy(np.concatenate(cols)).to(s.device)
        vals = torch.from_numpy(np.concatenate(vals)).to(s.device, torch.float32)
        personal = torch.zeros_like(s).index_put_((rows, cols), vals, accumulate=True)
        # every history track ranks above every fallback track
        return torch.where(personal > 0, 1 + personal, s)
