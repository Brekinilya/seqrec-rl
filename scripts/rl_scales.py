"""Spread of every reward under the base policy on training states.

Rewards are divided by this scale before optimisation, so one beta grid means the
same strength of regularisation for rewards measured in very different units.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
import torch

from seqrec_rl.data import N_SPECIAL, TemporalSplit
from seqrec_rl.model import load_model
from seqrec_rl.reward import load_rm, make_event_arrays
from seqrec_rl.reward_data import token_length_buckets
from seqrec_rl.rl import REWARDS, TrainStates, reward_scales


def train_event_arrays(data: Path, split: TemporalSplit, n_users: int | None, seed: int):
    ev = pl.read_parquet(data / "rm_events.parquet").filter(pl.col("timestamp") < split.train_end)
    if n_users:
        users = ev["uid"].unique().sort().sample(n_users, seed=seed).to_list()
        ev = ev.filter(pl.col("uid").is_in(users))
    return make_event_arrays(ev.sort("uid", "timestamp"), split.train_end)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed")
    p.add_argument("--base", default="checkpoints/base/best.pt")
    p.add_argument("--rm", default="checkpoints/rm/m0/best.pt")
    p.add_argument("--out", default="runs/rl/reward_scales.json")
    args = p.parse_args()
    data = Path(args.data)
    split = TemporalSplit.default()

    vocab = pl.read_parquet(data / "vocab.parquet")
    track_len = np.zeros(vocab.height + N_SPECIAL, dtype=np.float32)
    track_len[vocab["token"].to_numpy()] = vocab["track_length_seconds"].to_numpy()
    ev = train_event_arrays(data, split, 2000, seed=123)
    states = TrainStates(ev, 512, seed=123)
    base, _ = load_model(args.base)
    rm, _ = load_rm(args.rm, token_length_buckets(vocab))
    scales = reward_scales(states, base.eval(), rm, torch.as_tensor(track_len, device="cuda"), list(REWARDS))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scales, indent=2))
    print(json.dumps(scales, indent=2))


if __name__ == "__main__":
    main()
