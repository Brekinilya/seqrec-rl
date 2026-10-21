import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from .reward_data import HEADS


def labels_at(cols: dict, idx: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "ratio": np.clip(cols["ratio"][idx], 0, 100) / 100,
        "listen_plus": cols["lp"][idx].astype(np.float64),
        "like": cols["liked"][idx].astype(np.float64),
        "dislike": cols["disliked"][idx].astype(np.float64),
        "cont": cols["cont"][idx].astype(np.float64),
        "relisten": cols["relisten"][idx].astype(np.float64),
    }


def ece(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    which = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    err = 0.0
    for b in range(bins):
        m = which == b
        if m.any():
            err += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(err)


def soft_bce(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def head_metrics(head: str, y: np.ndarray, p: np.ndarray) -> dict:
    res = {"n": int(len(y)), "mean_label": float(y.mean()), "mean_pred": float(p.mean()), "bce": soft_bce(y, p)}
    if head in ("ratio", "cont"):
        res |= {"mae": float(np.abs(y - p).mean()), "rmse": float(np.sqrt(((y - p) ** 2).mean())),
                "spearman": float(spearmanr(y, p).statistic)}
    if head in ("listen_plus", "like", "dislike", "relisten") and 0 < y.sum() < len(y):
        res |= {"roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
                "ece": ece(y, p)}
    if head == "ratio":
        yb = (y >= 0.5).astype(np.float64)
        res["auc_listen_plus"] = float(roc_auc_score(yb, p))
    return res


def all_metrics(labels: dict, probs: np.ndarray, heads=HEADS, mask: np.ndarray | None = None) -> dict:
    out = {}
    for j, h in enumerate(heads):
        y, p = labels[h], probs[:, j]
        if mask is not None:
            y, p = y[mask], p[mask]
        if len(y):
            out[h] = head_metrics(h, y, p)
    return out
