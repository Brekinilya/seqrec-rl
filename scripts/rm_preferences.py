"""What the reward model would recommend if followed greedily (the beta -> 0 limit of KL-regularised RL).

For every test user the state is frozen at the test cutoff, the whole vocabulary
is scored, and the top-10 by each reward definition is compared with the base
GPT-2 and the baselines: relevance on test, list properties and the reward the
lists collect.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
import torch

from seqrec_rl.data import N_SPECIAL, TemporalSplit
from seqrec_rl.evaluation import ItemInfo, build_eval_set, list_stats, ranking_metrics
from seqrec_rl.reward import encode_last, history_mask, load_rm, make_event_arrays
from seqrec_rl.reward_data import HEADS, token_length_buckets

LISTS = ["GPT2", "PersonalRepeat", "MostPop"]


def rewards_from(probs: dict, track_len: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "ratio": probs["ratio"],
        "listen_plus": probs["listen_plus"],
        "seconds": probs["ratio"] * track_len,
        "like": probs["like"],
        "like_minus_dislike": probs["like"] - probs["dislike"],
        "like_plus_ratio": probs["like"] + 0.1 * probs["ratio"],
        "cont": probs["cont"],
        "relisten": probs["relisten"],
    }


def state_arrays(data: Path, uids: np.ndarray, cutoff: int, keep: int = 520):
    df = (pl.read_parquet(data / "rm_events.parquet")
          .filter(pl.col("uid").is_in(uids.tolist()) & (pl.col("timestamp") < cutoff))
          .sort("uid", "timestamp")
          .with_columns(pos=pl.int_range(pl.len(), dtype=pl.Int64).over("uid"),
                        n=pl.len().over("uid").cast(pl.Int64))
          .filter(pl.col("pos") >= pl.col("n") - keep).drop("pos", "n"))
    arrays = make_event_arrays(df, cutoff)
    return arrays, {u: i for i, u in enumerate(arrays.uids)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed")
    p.add_argument("--rm", default="checkpoints/rm/m0/best.pt")
    p.add_argument("--judges", nargs="*", default=["checkpoints/rm/m1/best.pt", "checkpoints/rm/m2/best.pt"])
    p.add_argument("--recs", default="runs/recs")
    p.add_argument("--out", default="reports/results")
    p.add_argument("--batch", type=int, default=48)
    args = p.parse_args()
    data, out = Path(args.data), Path(args.out)
    split = TemporalSplit.default()

    lis = pl.read_parquet(data / "listens.parquet")
    likes = pl.read_parquet(data / "likes.parquet")
    vocab = pl.read_parquet(data / "vocab.parquet")
    info = ItemInfo.from_vocab(vocab)
    n_tokens = len(info.train_count)
    es = build_eval_set(lis, likes, split, "test")
    del lis
    arrays, row_of = state_arrays(data, es.uids, split.val_end)
    model, _ = load_rm(args.rm, token_length_buckets(vocab))
    track_len = torch.as_tensor(info.track_length, dtype=torch.float32, device="cuda")

    given = {name: np.load(Path(args.recs) / f"recs_test_{name}.npz") for name in LISTS}
    for name, z in given.items():
        assert np.array_equal(z["uids"], es.uids), name
    judges = [load_rm(path, token_length_buckets(vocab))[0] for path in args.judges]
    names = list(rewards_from({h: torch.zeros(1, 1) for h in HEADS}, torch.zeros(1)).keys())
    greedy = {r: {"all": [], "new": []} for r in names}
    collected = {r: {k: [] for k in ["greedy"] + LISTS + ["targets"]} for r in names}
    judged = {r: {k: [] for k in ["greedy", "GPT2"]} for r in names}
    spread = {r: {k: [] for k in ["greedy", "GPT2"]} for r in names}

    def rewards_of(m, rows, hist):
        h = encode_last(m, arrays, rows, m.config.n_positions)
        with torch.no_grad():
            probs = {k: torch.sigmoid(v) for k, v in m.score_all(h, hist).items()}
        return rewards_from(probs, track_len)

    for start in range(0, len(es), args.batch):
        idx = np.arange(start, min(start + args.batch, len(es)))
        rows = np.array([row_of[es.uids[i]] for i in idx])
        hist = history_mask([es.history[i] for i in idx], n_tokens)
        proxy = rewards_of(model, rows, hist)
        judge_vals = [rewards_of(j, rows, hist) for j in judges]
        gpt_toks = torch.as_tensor(given["GPT2"]["all"][idx, :10], device="cuda")
        for r, val in proxy.items():
            val[:, :N_SPECIAL] = float("-inf")
            top_all = val.topk(100, dim=1).indices
            greedy[r]["all"].append(top_all.cpu().numpy())
            greedy[r]["new"].append(val.masked_fill(hist, float("-inf")).topk(100, dim=1).indices.cpu().numpy())
            collected[r]["greedy"].append(val.gather(1, top_all[:, :10]).mean(1).cpu().numpy())
            for name in LISTS:
                toks = torch.as_tensor(given[name]["all"][idx, :10], device="cuda")
                collected[r][name].append(val.gather(1, toks).mean(1).cpu().numpy())
            if judges:
                for key, toks in [("greedy", top_all[:, :10]), ("GPT2", gpt_toks)]:
                    each = torch.stack([v[r].gather(1, toks) for v in judge_vals])
                    judged[r][key].append(each.mean(0).mean(1).cpu().numpy())
                    stacked = torch.cat([val.gather(1, toks)[None], each])
                    spread[r][key].append(stacked.std(0).mean(1).cpu().numpy())
            tg = [es.targets[i] for i in idx]
            vals = val.cpu().numpy()
            collected[r]["targets"].append(np.array([vals[j, t].mean() if len(t) else np.nan
                                                     for j, t in enumerate(tg)]))

    results = {}
    for r in names:
        rec_all, rec_new = np.concatenate(greedy[r]["all"]), np.concatenate(greedy[r]["new"])
        results[r] = {
            "all": {"listen_plus": ranking_metrics(rec_all, es.targets, es.n_targets),
                    "lists": list_stats(rec_all, es, info)},
            "new": {"listen_plus": ranking_metrics(rec_new, es.new_targets, es.n_new_targets),
                    "lists": list_stats(rec_new, es, info)},
            "reward@10": {k: float(np.nanmean(np.concatenate(v))) for k, v in collected[r].items()},
        }
        if judges:
            results[r]["judge_reward@10"] = {k: float(np.concatenate(v).mean()) for k, v in judged[r].items()}
            results[r]["ensemble_std@10"] = {k: float(np.concatenate(v).mean()) for k, v in spread[r].items()}
        print(r, json.dumps({"ndcg@10": results[r]["all"]["listen_plus"]["ndcg@10"],
                             **{k: round(v, 3) for k, v in results[r]["all"]["lists"].items()
                                if k in ("in_history@10", "track_length_sec@10", "log10_train_count@10")},
                             "reward@10": results[r]["reward@10"],
                             "judge@10": results[r].get("judge_reward@10"),
                             "std@10": results[r].get("ensemble_std@10")}), flush=True)
    (out / "rm_preferences.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
