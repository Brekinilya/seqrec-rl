"""Evaluate a GPT-2 checkpoint with the shared protocol."""
import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from seqrec_rl.data import TemporalSplit
from seqrec_rl.evaluation import ItemInfo, build_eval_set, evaluate
from seqrec_rl.model import GPTRecommender, load_model


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/base/best.pt")
    p.add_argument("--name", default="GPT2")
    p.add_argument("--parts", nargs="+", default=["test"])
    p.add_argument("--data", default="data/processed")
    p.add_argument("--out", default="reports/results")
    p.add_argument("--recs-dir", default="runs/recs")
    args = p.parse_args()
    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    recs_dir = Path(args.recs_dir)
    recs_dir.mkdir(parents=True, exist_ok=True)

    lis = pl.read_parquet(data / "listens.parquet")
    likes = pl.read_parquet(data / "likes.parquet")
    info = ItemInfo.from_vocab(pl.read_parquet(data / "vocab.parquet"))
    split = TemporalSplit.default()
    model, ckpt = load_model(args.ckpt)

    for part in args.parts:
        es = build_eval_set(lis, likes, split, part)
        res, recs = evaluate(GPTRecommender(model), es, info)
        res = {"checkpoint": args.ckpt, "step": ckpt.get("step"), **res}
        (out / f"{args.name}_{part}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        np.savez_compressed(recs_dir / f"recs_{part}_{args.name}.npz", uids=es.uids, **recs)
        print(part, json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
