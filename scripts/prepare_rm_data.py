"""Reward-model features and labels for every listen, plus the users held out from reward-model training."""
import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from seqrec_rl.reward_data import build_rm_events


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed")
    p.add_argument("--holdout-share", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    data = Path(args.data)

    lis = pl.read_parquet(data / "listens.parquet").sort("uid", "timestamp")
    ev = build_rm_events(lis)
    ev.write_parquet(data / "rm_events.parquet")

    uids = np.sort(ev["uid"].unique().to_numpy())
    rng = np.random.default_rng(args.seed)
    holdout = np.sort(rng.choice(uids, int(len(uids) * args.holdout_share), replace=False))
    (data / "rm_holdout_users.json").write_text(json.dumps(holdout.tolist()), encoding="utf-8")

    summary = ev.select(
        pl.len().alias("n"),
        pl.col("lp").mean().alias("listen_plus"),
        pl.col("liked").mean().alias("like"),
        pl.col("disliked").mean().alias("dislike"),
        pl.col("cont").mean().alias("cont_mean"),
        (pl.col("cont") == 0).mean().alias("cont_zero_share"),
        pl.col("is_repeat").mean().alias("repeat"),
        ((pl.col("next_lp_ts") - pl.col("timestamp")) <= 7 * 86400).fill_null(False).mean().alias("relisten7d"),
    )
    print(summary)
    print("holdout users:", len(holdout))


if __name__ == "__main__":
    main()
