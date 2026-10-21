from dataclasses import dataclass
from pathlib import Path

import polars as pl

# Yambda timestamps are seconds from the dataset start, rounded to 5 s
MINUTE = 60
HOUR = 60 * MINUTE
DAY = 24 * HOUR

# "Listen+" threshold from the Yambda benchmark
LISTEN_PLUS_PCT = 50
SESSION_GAP = 30 * MINUTE


def load_events(path: str | Path) -> pl.DataFrame:
    df = pl.read_parquet(path)
    if df.schema["event_type"] != pl.Categorical:
        df = df.with_columns(pl.col("event_type").cast(pl.Utf8).cast(pl.Categorical))
    return df


def listens(events: pl.DataFrame) -> pl.DataFrame:
    return events.filter(pl.col("event_type") == "listen").drop("event_type")


@dataclass(frozen=True)
class TemporalSplit:
    """Global temporal split: train | gap | val | gap | test.

    Defaults follow the Yambda benchmark (test starts 30 min after day 300 and
    runs to the end of the data, ~22 h); validation is the last day of the
    benchmark's training period. Model state is frozen at the start of each
    evaluation window, so events inside a gap are neither history nor targets.
    """

    train_end: int
    val_start: int
    val_end: int
    test_start: int

    @classmethod
    def default(cls, train_days: int = 299, val_days: int = 1, gap: int = SESSION_GAP) -> "TemporalSplit":
        train_end = train_days * DAY
        val_end = train_end + val_days * DAY
        return cls(train_end, train_end + gap, val_end, val_end + gap)

    def train(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.filter(pl.col("timestamp") < self.train_end)

    def val(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.filter(pl.col("timestamp").is_between(self.val_start, self.val_end, closed="left"))

    def test(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.filter(pl.col("timestamp") >= self.test_start)

    def history_for_val(self, df: pl.DataFrame) -> pl.DataFrame:
        return self.train(df)

    def history_for_test(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.filter(pl.col("timestamp") < self.val_end)
