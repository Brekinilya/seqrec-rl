"""Per-event features and response labels for the reward model."""
import warnings

import numpy as np
import polars as pl

from .data import DAY, HOUR, LISTEN_PLUS_PCT, N_SPECIAL, SESSION_GAP

# input feature buckets; the last id of each is reserved for "unknown" (generated items)
RATIO_EDGES = [10, 30, 50, 80, 100, 101]               # 7 buckets
GAP_EDGES = [10, 30, 60, 120, 180, 240, 360, 600, 1800, HOUR, 6 * HOUR, DAY, 7 * DAY]  # 0 = first event, 1..14
LEN_EDGES = [90, 120, 150, 180, 210, 240, 300, 420, 600]  # 10 buckets
N_RATIO, N_GAP, N_LEN = len(RATIO_EDGES) + 2, len(GAP_EDGES) + 3, len(LEN_EDGES) + 1
UNKNOWN_RATIO, UNKNOWN_GAP = N_RATIO - 1, N_GAP - 1

CONT_GAMMA, CONT_HORIZON = 0.9, 20
CONT_NORM = (1 - CONT_GAMMA ** CONT_HORIZON) / (1 - CONT_GAMMA)
RELISTEN_MIN_GAP, RELISTEN_WINDOW = HOUR, 7 * DAY

HEADS = ("ratio", "listen_plus", "like", "dislike", "cont", "relisten")


def bucketize(x: np.ndarray, edges: list) -> np.ndarray:
    return np.searchsorted(np.asarray(edges), x, side="right").astype(np.int8)


def build_rm_events(lis: pl.DataFrame) -> pl.DataFrame:
    """lis: tokenized listens sorted by (uid, timestamp), with liked/disliked flags."""
    df = lis.with_columns(
        gap=pl.col("timestamp").diff().over("uid"),
        lp=pl.col("played_ratio_pct") >= LISTEN_PLUS_PCT,
        is_repeat=pl.int_range(pl.len()).over("uid", "item_id") > 0,
    ).with_columns(
        session=(pl.col("gap").is_null() | (pl.col("gap") > SESSION_GAP)).cum_sum().over("uid"),
    )

    # discounted number of Listen+ events later in the same session
    cont = pl.lit(0.0)
    for j in range(1, CONT_HORIZON + 1):
        same = pl.col("session").shift(-j).over("uid") == pl.col("session")
        nxt = pl.col("lp").shift(-j).over("uid")
        cont = cont + CONT_GAMMA ** (j - 1) * (same & nxt).fill_null(False).cast(pl.Float32)
    df = df.with_columns(cont=(cont / CONT_NORM).cast(pl.Float32))

    # time of the next Listen+ of the same track at least an hour later
    df = df.with_row_index("row")
    probe = (df.select("row", "uid", "item_id", (pl.col("timestamp") + RELISTEN_MIN_GAP).alias("probe_ts"))
             .sort("probe_ts"))
    later = (df.filter(pl.col("lp")).select("uid", "item_id", pl.col("timestamp").alias("next_lp_ts"))
             .sort("next_lp_ts"))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Sortedness of columns")
        nxt = probe.join_asof(later, left_on="probe_ts", right_on="next_lp_ts", by=["uid", "item_id"],
                              strategy="forward").select("row", "next_lp_ts")
    df = df.join(nxt, on="row", how="left").sort("row")

    gap = df["gap"].to_numpy()
    gap_bucket = np.where(np.isnan(gap.astype(np.float64)), 0, bucketize(np.nan_to_num(gap), GAP_EDGES) + 1)
    flags = (df["liked"].cast(pl.Int8) + 2 * df["disliked"].cast(pl.Int8)
             + 4 * df["is_organic"].cast(pl.Int8)).to_numpy()
    return df.select(
        "uid", "timestamp", "token", "is_organic", "played_ratio_pct", "liked", "disliked",
        "is_repeat", "lp", "cont", "next_lp_ts",
    ).with_columns(
        ratio_bucket=pl.Series(bucketize(df["played_ratio_pct"].to_numpy(), RATIO_EDGES)),
        gap_bucket=pl.Series(gap_bucket.astype(np.int8)),
        flags=pl.Series(flags.astype(np.int8)),
    )


def relisten_labels(ts: np.ndarray, next_lp_ts: np.ndarray, cutoff: int) -> tuple[np.ndarray, np.ndarray]:
    """Label and validity of 'listened to the track again within 7 days', observed only up to cutoff."""
    nxt = np.where(np.isnan(next_lp_ts), np.inf, next_lp_ts)
    label = nxt <= ts + RELISTEN_WINDOW
    valid = ts + RELISTEN_WINDOW < cutoff
    return label, valid


def length_buckets(track_length: np.ndarray) -> np.ndarray:
    return bucketize(track_length, LEN_EDGES)


def token_length_buckets(vocab: pl.DataFrame) -> np.ndarray:
    """Length bucket of every token (special tokens get bucket 0)."""
    lb = np.zeros(vocab.height + N_SPECIAL, dtype=np.int64)
    lb[vocab["token"].to_numpy()] = length_buckets(vocab["track_length_seconds"].to_numpy())
    return lb
