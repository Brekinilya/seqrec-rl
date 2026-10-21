import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

# Yambda timestamps are seconds from the dataset start, rounded to 5 s
MINUTE = 60
HOUR = 60 * MINUTE
DAY = 24 * HOUR
LAST_TIMESTAMP = 26_000_000

# "Listen+" threshold from the Yambda benchmark
LISTEN_PLUS_PCT = 50
SESSION_GAP = 30 * MINUTE

# a like/dislike counts for a listen if it comes within track length + this margin
FEEDBACK_MARGIN_SEC = 60

PAD, OOV = 0, 1
N_SPECIAL = 2


def load_events(path: str | Path) -> pl.DataFrame:
    df = pl.read_parquet(path)
    if df.schema["event_type"] != pl.Categorical:
        df = df.with_columns(pl.col("event_type").cast(pl.Utf8).cast(pl.Categorical))
    return df.with_columns(pl.col("timestamp").cast(pl.Int32))


def listens(events: pl.DataFrame) -> pl.DataFrame:
    return (events.filter(pl.col("event_type") == "listen").drop("event_type")
            .with_columns(pl.col("played_ratio_pct").cast(pl.Int32),
                          pl.col("track_length_seconds").cast(pl.Int32))
            .sort("uid", "timestamp"))


@dataclass(frozen=True)
class TemporalSplit:
    """Global temporal split: train | gap | val | gap | test.

    Boundaries match the Yambda benchmark code: test is the last 24 h of the
    data, validation is the preceding day, 30 min gaps in between. Model state
    is frozen at the start of each evaluation window, so events inside a gap
    are neither history nor targets.
    """

    train_end: int
    val_start: int
    val_end: int
    test_start: int

    @classmethod
    def default(cls, test_days: int = 1, val_days: int = 1, gap: int = SESSION_GAP) -> "TemporalSplit":
        test_start = LAST_TIMESTAMP - test_days * DAY
        val_end = test_start - gap
        val_start = val_end - val_days * DAY
        return cls(val_start - gap, val_start, val_end, test_start)

    def train(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.filter(pl.col("timestamp") < self.train_end)

    def val(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.filter(pl.col("timestamp").is_between(self.val_start, self.val_end, closed="left"))

    def test(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.filter(pl.col("timestamp") >= self.test_start)

    def window(self, df: pl.DataFrame, part: str) -> pl.DataFrame:
        return self.val(df) if part == "val" else self.test(df)

    def history(self, df: pl.DataFrame, part: str) -> pl.DataFrame:
        cutoff = self.train_end if part == "val" else self.val_end
        return df.filter(pl.col("timestamp") < cutoff)


def match_feedback(lis: pl.DataFrame, events: pl.DataFrame, kind: str) -> pl.DataFrame:
    """Match each like/dislike to the latest listen of the same track at or before it."""
    fb = (events.filter(pl.col("event_type") == kind)
          .select("uid", "item_id", pl.col("timestamp").alias("fb_ts"))
          .sort("fb_ts"))
    right = lis.select("uid", "item_id", "timestamp", "track_length_seconds").sort("timestamp")
    with warnings.catch_warnings():
        # both sides are sorted by time globally, which is what join_asof needs
        warnings.filterwarnings("ignore", message="Sortedness of columns")
        m = fb.join_asof(right, left_on="fb_ts", right_on="timestamp", by=["uid", "item_id"],
                         strategy="backward")
    return m.with_columns(delay_sec=pl.col("fb_ts") - pl.col("timestamp"))


def flag_listens(lis: pl.DataFrame, matched: pl.DataFrame, name: str) -> pl.DataFrame:
    hits = (matched.filter(pl.col("timestamp").is_not_null()
                           & (pl.col("delay_sec") <= pl.col("track_length_seconds") + FEEDBACK_MARGIN_SEC))
            .select("uid", "item_id", "timestamp").unique()
            .with_columns(pl.lit(True).alias(name)))
    return (lis.join(hits, on=["uid", "item_id", "timestamp"], how="left", maintain_order="left")
            .with_columns(pl.col(name).fill_null(False)))


def attach_feedback(lis: pl.DataFrame, events: pl.DataFrame) -> pl.DataFrame:
    lis = flag_listens(lis, match_feedback(lis, events, "like"), "liked")
    return flag_listens(lis, match_feedback(lis, events, "dislike"), "disliked")


def build_vocab(train_lis: pl.DataFrame, min_count: int) -> pl.DataFrame:
    """Tokens N_SPECIAL.. are assigned to tracks by decreasing train popularity."""
    counts = (train_lis.group_by("item_id")
              .agg(train_count=pl.len(), track_length_seconds=pl.col("track_length_seconds").first())
              .filter(pl.col("train_count") >= min_count)
              .sort("train_count", "item_id", descending=[True, False]))
    return counts.with_columns(token=pl.int_range(N_SPECIAL, N_SPECIAL + counts.height, dtype=pl.Int32))


def tokenize(df: pl.DataFrame, vocab: pl.DataFrame) -> pl.DataFrame:
    return (df.join(vocab.select("item_id", "token"), on="item_id", how="left", maintain_order="left")
            .with_columns(pl.col("token").fill_null(OOV)))


def user_sequences(df: pl.DataFrame, col: str = "token") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenated per-user sequences: (uids, offsets, values), user i owns values[offsets[i]:offsets[i+1]].

    Expects df sorted by (uid, timestamp).
    """
    lens = df.group_by("uid", maintain_order=True).len()
    offsets = np.zeros(lens.height + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lens["len"].to_numpy())
    return lens["uid"].to_numpy(), offsets, df[col].to_numpy()
