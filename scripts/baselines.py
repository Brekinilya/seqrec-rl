"""Tune MostPop and PersonalRepeat on validation, evaluate the chosen settings on test."""
import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from seqrec_rl.baselines import MostPop, PersonalRepeat
from seqrec_rl.data import TemporalSplit
from seqrec_rl.evaluation import ItemInfo, build_eval_set, evaluate

POP_DAYS = [1, 7, 30]
GAMMAS = [0.99, 0.998, 0.9995, 1.0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--out", default="reports/results")
    parser.add_argument("--recs-dir", default="runs/recs")
    args = parser.parse_args()
    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    recs_dir = Path(args.recs_dir)
    recs_dir.mkdir(parents=True, exist_ok=True)

    lis = pl.read_parquet(data / "listens.parquet")
    likes = pl.read_parquet(data / "likes.parquet")
    vocab = pl.read_parquet(data / "vocab.parquet")
    info = ItemInfo.from_vocab(vocab)
    n_tokens = len(info.train_count)
    split = TemporalSplit.default()

    val = build_eval_set(lis, likes, split, "val")
    grid = []
    for days in POP_DAYS:
        res, _ = evaluate(MostPop(lis, split, "val", n_tokens, days), val, info)
        grid.append({"model": "MostPop", "days": days, **res["all"]["listen_plus"]})
        for gamma in GAMMAS:
            for lp in (False, True):
                rec = PersonalRepeat(MostPop(lis, split, "val", n_tokens, days), gamma, lp)
                res, _ = evaluate(rec, val, info)
                grid.append({"model": "PersonalRepeat", "days": days, "gamma": gamma,
                             "listen_plus_only": lp, **res["all"]["listen_plus"]})
        print(grid[-1])
    grid_df = pl.DataFrame(grid)
    print(grid_df.sort("ndcg@10", descending=True).head(10))
    (out / "baselines_val_grid.json").write_text(json.dumps(grid, indent=2), encoding="utf-8")

    best_pop = grid_df.filter(pl.col("model") == "MostPop").sort("ndcg@10", descending=True).row(0, named=True)
    best_rep = grid_df.filter(pl.col("model") == "PersonalRepeat").sort("ndcg@10", descending=True).row(0, named=True)

    test = build_eval_set(lis, likes, split, "test")
    pop = MostPop(lis, split, "test", n_tokens, best_pop["days"])
    rep = PersonalRepeat(MostPop(lis, split, "test", n_tokens, best_rep["days"]),
                         best_rep["gamma"], best_rep["listen_plus_only"])
    results = {}
    for name, rec, params in [("MostPop", pop, {"days": best_pop["days"]}),
                              ("PersonalRepeat", rep, {k: best_rep[k] for k in ("days", "gamma", "listen_plus_only")})]:
        res, recs = evaluate(rec, test, info)
        results[name] = {"params": params, **res}
        np.savez_compressed(recs_dir / f"recs_test_{name}.npz", uids=test.uids, **recs)
        print(name, json.dumps(res, indent=1))
    (out / "baselines_test.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
