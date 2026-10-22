"""Non-sequential baselines for the response prediction task.

Aggregates (item / user means) come from events before a cutoff; per-event
features look only at the user's own past, so nothing leaks from the target.
"""
import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from .data import N_SPECIAL
from .reward_data import HEADS, RELISTEN_WINDOW

LABEL_EXPR = {
    "ratio": pl.col("played_ratio_pct").clip(0, 100) / 100,
    "listen_plus": pl.col("lp").cast(pl.Float64),
    "like": pl.col("liked").cast(pl.Float64),
    "dislike": pl.col("disliked").cast(pl.Float64),
    "cont": pl.col("cont").cast(pl.Float64),
    "relisten": ((pl.col("next_lp_ts") - pl.col("timestamp")) <= RELISTEN_WINDOW).fill_null(False).cast(pl.Float64),
}
SMOOTH = 20


def with_labels(ev: pl.DataFrame) -> pl.DataFrame:
    return ev.with_columns(**{f"y_{h}": e for h, e in LABEL_EXPR.items()})


def causal_features(ev: pl.DataFrame) -> pl.DataFrame:
    """ev sorted by (uid, timestamp)."""
    return ev.with_columns(
        n_prev=pl.int_range(pl.len()).over("uid", "token"),
        since_last_h=((pl.col("timestamp") - pl.col("timestamp").shift(1).over("uid", "token")) / 3600)
        .log1p().fill_null(-1.0),
        prev_ratio=pl.col("played_ratio_pct").shift(1).over("uid").fill_null(-1),
        recent_lp=pl.col("lp").cast(pl.Float64).shift(1).rolling_mean(20, min_samples=1).over("uid").fill_null(-1.0),
    )


def aggregate_stats(ev: pl.DataFrame, cutoff: int) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """Smoothed item and user means of every label from events before cutoff."""
    past = with_labels(ev.filter(pl.col("timestamp") < cutoff))
    valid_relisten = pl.col("timestamp") + RELISTEN_WINDOW < cutoff
    glob = {h: past[f"y_{h}"].mean() for h in HEADS if h != "relisten"}
    glob["relisten"] = past.filter(valid_relisten)["y_relisten"].mean()

    def agg(key, prefix):
        exprs = [pl.len().alias(f"{prefix}_n")]
        for h in HEADS:
            y = pl.col(f"y_{h}")
            if h == "relisten":
                s, n = y.filter(valid_relisten).sum(), valid_relisten.sum()
            else:
                s, n = y.sum(), pl.len()
            exprs.append(((s + SMOOTH * glob[h]) / (n + SMOOTH)).alias(f"{prefix}_{h}"))
        return past.group_by(key).agg(exprs)

    return agg("token", "item"), agg("uid", "user"), glob


def feature_frame(rows: pl.DataFrame, items: pl.DataFrame, users: pl.DataFrame, glob: dict,
                  track_len: np.ndarray) -> pl.DataFrame:
    f = rows.join(items, on="token", how="left", maintain_order="left") \
            .join(users, on="uid", how="left", maintain_order="left")
    fill = {f"{p}_{h}": glob[h] for p in ("item", "user") for h in HEADS} | {"item_n": 0, "user_n": 0}
    f = f.with_columns([pl.col(c).fill_null(v) for c, v in fill.items()])
    return f.with_columns(track_len=pl.Series(track_len[f["token"].to_numpy()]))


FEATURES = (["is_repeat", "is_organic", "n_prev", "since_last_h", "prev_ratio", "recent_lp", "track_len",
             "item_n", "user_n"] + [f"item_{h}" for h in HEADS] + [f"user_{h}" for h in HEADS])


def to_matrix(f: pl.DataFrame) -> np.ndarray:
    x = f.select([pl.col(c).cast(pl.Float64) for c in FEATURES]).to_numpy()
    x[:, FEATURES.index("item_n")] = np.log1p(x[:, FEATURES.index("item_n")])
    x[:, FEATURES.index("user_n")] = np.log1p(x[:, FEATURES.index("user_n")])
    return x


def fit_gbm(f: pl.DataFrame, seed: int = 0) -> dict:
    x = to_matrix(f)
    models = {}
    for h in HEADS:
        y = f[f"y_{h}"].to_numpy()
        keep = np.ones(len(y), dtype=bool) if h != "relisten" else f["relisten_valid"].to_numpy()
        if h in ("ratio", "cont"):
            m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, random_state=seed)
        else:
            m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=seed)
        models[h] = m.fit(x[keep], y[keep])
    return models


def predict_gbm(models: dict, f: pl.DataFrame) -> np.ndarray:
    x = to_matrix(f)
    cols = []
    for h in HEADS:
        m = models[h]
        p = m.predict(x) if h in ("ratio", "cont") else m.predict_proba(x)[:, 1]
        cols.append(np.clip(p, 1e-4, 1 - 1e-4))
    return np.stack(cols, 1)


def mean_predictions(f: pl.DataFrame, glob: dict) -> dict[str, np.ndarray]:
    n = f.height
    out = {"global": np.stack([np.full(n, glob[h]) for h in HEADS], 1),
           "item": np.stack([f[f"item_{h}"].to_numpy() for h in HEADS], 1),
           "user": np.stack([f[f"user_{h}"].to_numpy() for h in HEADS], 1)}
    return {k: np.clip(v, 1e-4, 1 - 1e-4) for k, v in out.items()}


def in_vocab(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(pl.col("token") >= N_SPECIAL)
