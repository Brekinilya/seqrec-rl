"""Paired bootstrap over test users for the key relevance comparisons against the base GPT-2."""
import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
import torch

from seqrec_rl.data import TemporalSplit
from seqrec_rl.evaluation import ItemInfo, build_eval_set, per_user_metric
from seqrec_rl.model import load_model
from seqrec_rl.reward import history_mask, load_rm
from seqrec_rl.reward_data import token_length_buckets
from seqrec_rl.rl import NEG, cutoff_states, encode_windows, pi_star_logp, policy_logits, rm_rewards

POLICIES = {
    "π* ratio β=1": ("pistar", "ratio", 1.0),
    "π* relisten β=1": ("pistar", "relisten", 1.0),
    "π* like β=1": ("pistar", "like", 1.0),
    "RL exact ratio β=1": ("model", "checkpoints/rl/ratio_b1_exact/policy.pt"),
    "Positive-CE": ("model", "checkpoints/rl/positive_ce/policy.pt"),
}
METRICS = [("all", "ndcg", 10), ("all", "recall", 100), ("new", "ndcg", 10)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed")
    p.add_argument("--base", default="checkpoints/base/best.pt")
    p.add_argument("--rm", default="checkpoints/rm/m0/best.pt")
    p.add_argument("--scales", default="runs/rl/reward_scales.json")
    p.add_argument("--recs", default="runs/recs")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--out", default="reports/results/bootstrap_ci.json")
    args = p.parse_args()
    data = Path(args.data)
    split = TemporalSplit.default()

    lis = pl.read_parquet(data / "listens.parquet")
    likes = pl.read_parquet(data / "likes.parquet")
    vocab = pl.read_parquet(data / "vocab.parquet")
    info = ItemInfo.from_vocab(vocab)
    n_tokens = len(info.train_count)
    es = build_eval_set(lis, likes, split, "test")
    del lis
    arrays, row_of = cutoff_states(data / "rm_events.parquet", es.uids, split.val_end)
    track_len = torch.as_tensor(info.track_length, dtype=torch.float32, device="cuda")
    rm = load_rm(args.rm, token_length_buckets(vocab))[0]
    base = load_model(args.base)[0].eval()
    models = {n: load_model(spec[1])[0].eval() for n, spec in POLICIES.items() if spec[0] == "model"}
    scales = json.loads(Path(args.scales).read_text())

    recs = {n: {"all": [], "new": []} for n in ["GPT-2", *POLICIES]}
    for start in range(0, len(es), args.batch):
        idx = np.arange(start, min(start + args.batch, len(es)))
        contexts = [es.hist_tokens[i][-512:].astype(np.int64) for i in idx]
        hist = history_mask([es.history[i] for i in idx], n_tokens)
        rows = np.array([row_of[int(es.uids[i])] for i in idx])
        windows = np.stack([arrays.offsets[rows], arrays.offsets[rows + 1]], 1)
        with torch.no_grad():
            ref = torch.log_softmax(policy_logits(base, contexts), 1)
            rw = rm_rewards(rm, encode_windows(rm, arrays, windows), hist, track_len, ["ratio", "relisten", "like"])
            logps = {"GPT-2": ref}
            for n, spec in POLICIES.items():
                if spec[0] == "pistar":
                    logps[n] = pi_star_logp(ref, rw[spec[1]] / scales[spec[1]], spec[2])
                else:
                    logps[n] = torch.log_softmax(policy_logits(models[n], contexts), 1)
            for n, lp in logps.items():
                recs[n]["all"].append(lp.topk(100, 1).indices.cpu().numpy())
                recs[n]["new"].append(lp.masked_fill(hist, NEG).topk(100, 1).indices.cpu().numpy())
    for n in recs:
        recs[n] = {k: np.concatenate(v) for k, v in recs[n].items()}
    for name in ("PersonalRepeat", "MostPop"):
        z = np.load(Path(args.recs) / f"recs_test_{name}.npz")
        assert np.array_equal(z["uids"], es.uids)
        recs[name] = {"all": z["all"], "new": z["new"]}

    def values(n, proto, metric, k):
        tg, nt = (es.targets, es.n_targets) if proto == "all" else (es.new_targets, es.n_new_targets)
        return per_user_metric(recs[n][proto], tg, nt, k, metric)

    rng = np.random.default_rng(0)
    results = {}
    for proto, metric, k in METRICS:
        key = f"{proto}:{metric}@{k}"
        base_v = values("GPT-2", proto, metric, k)
        users = np.flatnonzero(~np.isnan(base_v))
        boots = rng.integers(0, len(users), (args.n_boot, len(users)))
        results[key] = {"GPT-2": {"mean": float(base_v[users].mean())}}
        for n in recs:
            if n == "GPT-2":
                continue
            v = values(n, proto, metric, k)
            d = v[users] - base_v[users]
            bd = d[boots].mean(1)
            lo, hi = np.percentile(bd, [2.5, 97.5])
            results[key][n] = {"mean": float(v[users].mean()), "diff": float(d.mean()),
                               "ci95": [float(lo), float(hi)], "rel": float(d.mean() / base_v[users].mean())}
        print(key, json.dumps({n: (round(r["mean"], 4), [round(x, 4) for x in r.get("ci95", [])])
                               for n, r in results[key].items()}, ensure_ascii=False), flush=True)
    Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
