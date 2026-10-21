"""Evaluation protocol shared by every model in the project.

For each user with targets in the evaluation window we rank the whole vocabulary
once, from the state frozen at the window start, and compare the top-K list with
the set of tracks the user interacted with in the window.

Two protocols:
  all  - nothing is filtered, targets are all Listen+ tracks (and liked tracks);
  new  - tracks from the user's history are removed from the ranking,
         targets are Listen+ tracks the user has never listened to before.

Targets outside the vocabulary stay in the denominators, so every model is
penalised for them in the same way.
"""
from dataclasses import dataclass

import numpy as np
import polars as pl
import torch

from .data import LISTEN_PLUS_PCT, N_SPECIAL, TemporalSplit


@dataclass
class EvalSet:
    part: str
    uids: np.ndarray
    hist_tokens: list[np.ndarray]   # full history, chronological, OOV included
    hist_ratio: list[np.ndarray]
    history: list[np.ndarray]       # unique in-vocabulary tokens of the history
    targets: list[np.ndarray]       # unique in-vocabulary Listen+ target tokens
    n_targets: np.ndarray           # unique Listen+ target tracks, OOV included
    new_targets: list[np.ndarray]
    n_new_targets: np.ndarray
    like_targets: list[np.ndarray]
    n_like_targets: np.ndarray

    def __len__(self) -> int:
        return len(self.uids)

    def contexts(self, idx, max_len: int) -> list[np.ndarray]:
        return [self.hist_tokens[i][-max_len:] for i in idx]


def _per_user(df: pl.DataFrame, uids: np.ndarray, cols: list[str]) -> dict[str, list[np.ndarray]]:
    g = df.group_by("uid", maintain_order=True).agg(cols)
    g = pl.DataFrame({"uid": uids}).join(g, on="uid", how="left", maintain_order="left")
    out = {}
    for c in cols:
        out[c] = [np.asarray(v if v is not None else [], dtype=np.int64) for v in g[c].to_list()]
    return out


def build_eval_set(lis: pl.DataFrame, likes: pl.DataFrame, split: TemporalSplit, part: str) -> EvalSet:
    hist = split.history(lis, part).sort("uid", "timestamp")
    window = split.window(lis, part)
    seen = hist.select("uid", "item_id").unique().with_columns(seen=pl.lit(True))

    tgt = (window.filter(pl.col("played_ratio_pct") >= LISTEN_PLUS_PCT)
           .select("uid", "item_id", "token").unique(["uid", "item_id"])
           .join(seen, on=["uid", "item_id"], how="left").with_columns(pl.col("seen").fill_null(False)))
    like_tgt = split.window(likes, part).select("uid", "item_id", "token").unique(["uid", "item_id"])

    users = pl.concat([tgt.select("uid"), like_tgt.select("uid")]).unique()
    users = users.join(hist.select("uid").unique(), on="uid").sort("uid")["uid"].to_numpy()

    h = _per_user(hist, users, ["token", "played_ratio_pct"])

    def targets(df):
        counts = pl.DataFrame({"uid": users}).join(df.group_by("uid").len(), on="uid", how="left")
        toks = _per_user(df.filter(pl.col("token") >= N_SPECIAL).sort("uid"), users, ["token"])["token"]
        return toks, counts["len"].fill_null(0).to_numpy()

    t, nt = targets(tgt)
    nw, nnw = targets(tgt.filter(~pl.col("seen")))
    lk, nlk = targets(like_tgt)
    history = [np.unique(x[x >= N_SPECIAL]) for x in h["token"]]
    return EvalSet(part, users, h["token"], h["played_ratio_pct"], history, t, nt, nw, nnw, lk, nlk)


@dataclass
class ItemInfo:
    """Per-token item attributes; special tokens hold zeros."""
    train_count: np.ndarray
    track_length: np.ndarray
    artist: np.ndarray
    n_items: int

    @classmethod
    def from_vocab(cls, vocab: pl.DataFrame) -> "ItemInfo":
        n = vocab.height + N_SPECIAL

        def col(name):
            a = np.zeros(n, dtype=np.int64)
            a[vocab["token"].to_numpy()] = vocab[name].fill_null(0).to_numpy()
            return a

        return cls(col("train_count"), col("track_length_seconds"), col("artist_id"), vocab.height)


def top_k(rec, es: EvalSet, k: int, exclude_history: bool, batch_size: int = 256) -> np.ndarray:
    """rec.scores(es, idx) -> (len(idx), n_tokens) tensor; returns top-k tokens per user."""
    out = []
    for start in range(0, len(es), batch_size):
        idx = np.arange(start, min(start + batch_size, len(es)))
        s = rec.scores(es, idx)
        s[:, :N_SPECIAL] = float("-inf")
        if exclude_history:
            rows = np.concatenate([np.full(len(es.history[i]), j) for j, i in enumerate(idx)])
            cols = np.concatenate([es.history[i] for i in idx])
            s[torch.from_numpy(rows).to(s.device), torch.from_numpy(cols).to(s.device)] = float("-inf")
        out.append(s.topk(k, dim=1).indices.cpu().numpy())
    return np.concatenate(out)


def ranking_metrics(recs: np.ndarray, targets: list[np.ndarray], n_targets: np.ndarray,
                    ks=(10, 100)) -> dict:
    users = np.flatnonzero(n_targets > 0)
    K = max(ks)
    hits = np.zeros((len(users), K), dtype=bool)
    for row, i in enumerate(users):
        hits[row] = np.isin(recs[i, :K], targets[i])
    n = n_targets[users]
    disc = 1.0 / np.log2(np.arange(2, K + 2))
    res = {"n_users": int(len(users))}
    for k in ks:
        h = hits[:, :k]
        denom = np.minimum(n, k)
        dcg = (h * disc[:k]).sum(1)
        idcg = np.cumsum(disc)[denom - 1]
        prec = np.cumsum(h, 1) / np.arange(1, k + 1)
        res[f"recall@{k}"] = float((h.sum(1) / denom).mean())
        # hits / |T|, the definition behind the numbers in the Yambda paper
        res[f"recall_raw@{k}"] = float((h.sum(1) / n).mean())
        res[f"ndcg@{k}"] = float((dcg / idcg).mean())
        res[f"map@{k}"] = float(((prec * h).sum(1) / denom).mean())
        res[f"hitrate@{k}"] = float(h.any(1).mean())
    return res


def gini(counts: np.ndarray) -> float:
    x = np.sort(counts.astype(np.float64))
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def list_stats(recs: np.ndarray, es: EvalSet, info: ItemInfo, k: int = 10, n_pairs: int = 5000,
               seed: int = 0) -> dict:
    """Beyond-accuracy properties of the top-k lists."""
    r = recs[:, :k]
    rng = np.random.default_rng(seed)
    a, b = rng.integers(0, len(r), n_pairs), rng.integers(0, len(r), n_pairs)
    keep = a != b
    jac = [len(np.intersect1d(r[i], r[j])) / len(np.union1d(r[i], r[j])) for i, j in zip(a[keep], b[keep])]
    exposure = np.bincount(r.ravel(), minlength=len(info.train_count))[N_SPECIAL:]
    top1 = np.bincount(recs[:, 0])
    return {
        f"coverage@{k}": float(len(np.unique(r)) / info.n_items),
        f"coverage@{recs.shape[1]}": float(len(np.unique(recs)) / info.n_items),
        "gini@10": gini(exposure),
        "pair_jaccard@10": float(np.mean(jac)),
        "top1_most_common_share": float(top1.max() / len(recs)),
        "log10_train_count@10": float(np.log10(info.train_count[r]).mean()),
        "track_length_sec@10": float(info.track_length[r].mean()),
        "short_share@10": float((info.track_length[r] < 150).mean()),
        "in_history@10": float(np.mean([np.isin(r[i], es.history[i]).mean() for i in range(len(r))])),
        "unique_artists@10": float(np.mean([len(np.unique(info.artist[row])) for row in r])),
    }


def evaluate(rec, es: EvalSet, info: ItemInfo, ks=(10, 100), batch_size: int = 256) -> tuple[dict, dict]:
    K = max(ks)
    res, recs = {}, {}
    for proto, exclude in [("all", False), ("new", True)]:
        r = top_k(rec, es, K, exclude, batch_size)
        recs[proto] = r
        res[proto] = {"lists": list_stats(r, es, info)}
        if proto == "all":
            res[proto]["listen_plus"] = ranking_metrics(r, es.targets, es.n_targets, ks)
            res[proto]["like"] = ranking_metrics(r, es.like_targets, es.n_like_targets, ks)
        else:
            res[proto]["listen_plus"] = ranking_metrics(r, es.new_targets, es.n_new_targets, ks)
    return res, recs
