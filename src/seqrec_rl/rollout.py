"""Multi-step generation: generated tracks are appended to the context as if the user had listened to them.

The policy only sees track ids, so appending a token is enough. The reward model
also sees the response to every past track; for a generated track that response
never happened and has to be assumed:

  unknown     the "unknown" response and gap tokens the reward model was trained with
  expected    the played-ratio bucket the proxy reward model itself predicts
  optimistic  a full listen
"""
import numpy as np
import torch

from .data import PAD
from .reward import RewardModel
from .reward_data import GAP_EDGES, RATIO_EDGES, UNKNOWN_GAP, UNKNOWN_RATIO, bucketize

FEEDBACK_MODES = ("unknown", "expected", "optimistic")
TYPICAL_GAP = int(bucketize(np.array([180]), GAP_EDGES)[0]) + 1  # consecutive tracks are ~3 min apart
FULL_LISTEN = int(bucketize(np.array([100]), RATIO_EDGES)[0])
COLS = ("token", "ratio_bucket", "flags", "gap_bucket")
PAD_VALUES = {"token": PAD, "ratio_bucket": UNKNOWN_RATIO, "flags": 0, "gap_bucket": UNKNOWN_GAP}


def ratio_bucket(pred_ratio: np.ndarray) -> np.ndarray:
    return bucketize(np.clip(pred_ratio, 0, 1) * 100, RATIO_EDGES)


class RMSeq:
    """Per-user reward-model input sequences that can be extended one event at a time."""

    def __init__(self, cols: dict[str, list[np.ndarray]]):
        self.cols = cols

    @classmethod
    def from_events(cls, ev, starts: np.ndarray, ends: np.ndarray, keep: int = 512) -> "RMSeq":
        return cls({c: [ev.cols[c][max(s, e - keep):e].astype(np.int64) for s, e in zip(starts, ends)]
                    for c in COLS})

    def copy(self) -> "RMSeq":
        return RMSeq({c: list(v) for c, v in self.cols.items()})

    def append(self, **values: np.ndarray) -> None:
        for c in COLS:
            self.cols[c] = [np.append(x, v) for x, v in zip(self.cols[c], values[c])]

    def generated(self, tokens: np.ndarray, mode: str, pred_ratio: np.ndarray | None = None) -> None:
        n = len(tokens)
        if mode == "unknown":
            rb, gap = np.full(n, UNKNOWN_RATIO), np.full(n, UNKNOWN_GAP)
        elif mode == "expected":
            rb, gap = ratio_bucket(pred_ratio), np.full(n, TYPICAL_GAP)
        else:
            rb, gap = np.full(n, FULL_LISTEN), np.full(n, TYPICAL_GAP)
        self.append(token=tokens, ratio_bucket=rb, flags=np.zeros(n, np.int64), gap_bucket=gap)

    @torch.no_grad()
    def encode(self, model: RewardModel) -> torch.Tensor:
        L = model.config.n_positions
        seqs = {c: [x[-L:] for x in v] for c, v in self.cols.items()}
        lengths = np.array([len(x) for x in seqs["token"]])
        dev = next(model.parameters()).device
        t = {}
        for c in COLS:
            buf = np.full((len(lengths), lengths.max()), PAD_VALUES[c], dtype=np.int64)
            for i, x in enumerate(seqs[c]):
                buf[i, :len(x)] = x
            t[c] = torch.from_numpy(buf).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model.encode(t["token"], t["ratio_bucket"], t["flags"], t["gap_bucket"])
        return h[torch.arange(len(lengths), device=dev), torch.as_tensor(lengths - 1, device=dev)].float()


@torch.no_grad()
def item_probs(model: RewardModel, h: torch.Tensor, items: torch.Tensor, rep: torch.Tensor) -> torch.Tensor:
    """(B, H) response probabilities for one candidate per state, shown as a recommendation."""
    return torch.sigmoid(model.logits(h, items, rep, torch.zeros_like(items)).float())
