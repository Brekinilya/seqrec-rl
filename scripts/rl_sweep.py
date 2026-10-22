"""Evaluate KL-regularised policies on the test states.

Policies are either the closed-form optimum pi* ~ pi_ref exp(r / beta) for a grid
of rewards, defences and betas, or RL-trained checkpoints. Every policy is scored
with the shared relevance protocol, list statistics, KL to the base model and the
reward it collects according to the proxy (m0) and to the judges (m1, m2).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
import torch

from seqrec_rl.data import TemporalSplit
from seqrec_rl.evaluation import ItemInfo, build_eval_set, list_stats, ranking_metrics
from seqrec_rl.model import load_model
from seqrec_rl.reward import history_mask, load_rm
from seqrec_rl.reward_data import token_length_buckets
from seqrec_rl.rl import (NEG, REWARDS, cutoff_states, encode_windows, kl, pi_star_logp,
                          policy_logits, rm_rewards)

BETAS = [3, 1, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02]


def config_grid() -> list[dict]:
    grid = [{"name": "base", "reward": None}]
    for r in REWARDS:
        for b in BETAS:
            grid.append({"name": f"pistar/{r}/b{b}", "reward": r, "beta": b, "defence": "none"})
    for r in ("ratio", "like", "like_plus_ratio"):
        for b in BETAS:
            for kappa in (1, 3):
                grid.append({"name": f"pistar/{r}/pess{kappa}/b{b}", "reward": r, "beta": b,
                             "defence": "pess", "kappa": kappa})
            grid.append({"name": f"pistar/{r}/top200/b{b}", "reward": r, "beta": b, "defence": "top200"})
    return grid


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed")
    p.add_argument("--base", default="checkpoints/base/best.pt")
    p.add_argument("--rm-dir", default="checkpoints/rm")
    p.add_argument("--scales", default="runs/rl/reward_scales.json")
    p.add_argument("--policies", nargs="*", default=[], help="name=checkpoint of RL-trained policies")
    p.add_argument("--no-grid", action="store_true", help="only evaluate --policies (and the base)")
    p.add_argument("--out", default="reports/results/rl_sweep.json")
    p.add_argument("--batch", type=int, default=32)
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
    lb = token_length_buckets(vocab)
    rms = {m: load_rm(Path(args.rm_dir) / m / "best.pt", lb)[0] for m in ("m0", "scratch10", "m1", "m2")}
    base, _ = load_model(args.base)
    base.eval()
    scales = json.loads(Path(args.scales).read_text())
    policies = {}
    for spec in args.policies:
        name, path = spec.split("=")
        policies[name] = load_model(path)[0].eval()

    grid = ([] if args.no_grid else config_grid())
    grid = [c for c in grid if c["reward"] is not None] + [{"name": "base", "reward": None}]
    grid += [{"name": n, "reward": None, "policy": n} for n in policies]
    names = list(REWARDS)
    acc = {c["name"]: {"all": [], "new": [], "kl": [], **{f"E_{w}_{r}": [] for w in ("proxy", "judge") for r in names},
                       **{f"top10_{w}_{r}": [] for w in ("proxy", "judge") for r in names}} for c in grid}

    for start in range(0, len(es), args.batch):
        idx = np.arange(start, min(start + args.batch, len(es)))
        contexts = [es.hist_tokens[i][-base.config.n_positions:].astype(np.int64) for i in idx]
        with torch.no_grad():
            ref_logp = torch.log_softmax(policy_logits(base, contexts), 1)
            pol_logp = {n: torch.log_softmax(policy_logits(m, contexts), 1) for n, m in policies.items()}
        hist = history_mask([es.history[i] for i in idx], n_tokens)
        rows = np.array([row_of[int(es.uids[i])] for i in idx])
        windows = np.stack([arrays.offsets[rows], arrays.offsets[rows + 1]], 1)
        rw = {m: rm_rewards(model, encode_windows(model, arrays, windows), hist, track_len, names)
              for m, model in rms.items()}
        judge = {n: (rw["m1"][n] + rw["m2"][n]) / 2 for n in names}

        for c in grid:
            with torch.no_grad():
                if c.get("policy"):
                    logp = pol_logp[c["policy"]]
                elif c["reward"] is None:
                    logp = ref_logp
                else:
                    r = rw["m0"][c["reward"]]
                    cand = None
                    if c["defence"] == "pess":
                        r = r - c["kappa"] * (r - rw["scratch10"][c["reward"]]).abs() / 2
                    elif c["defence"] == "top200":
                        thr = ref_logp.topk(200, dim=1).values[:, -1:]
                        cand = ref_logp >= thr
                    logp = pi_star_logp(ref_logp, r / scales[c["reward"]], c["beta"], cand)
                a = acc[c["name"]]
                top = logp.topk(100, dim=1).indices
                a["all"].append(top.cpu().numpy().astype(np.int32))
                a["new"].append(logp.masked_fill(hist, NEG).topk(100, dim=1).indices.cpu().numpy().astype(np.int32))
                a["kl"].append(kl(logp, ref_logp).cpu().numpy())
                prob = logp.exp()
                for n in names:
                    for w, vals in (("proxy", rw["m0"][n]), ("judge", judge[n])):
                        a[f"E_{w}_{n}"].append((prob * vals).sum(1).cpu().numpy())
                        a[f"top10_{w}_{n}"].append(vals.gather(1, top[:, :10]).mean(1).cpu().numpy())
        print(f"{start + len(idx)}/{len(es)}", flush=True)

    results = []
    for c in grid:
        a = acc[c["name"]]
        rec_all, rec_new = np.concatenate(a["all"]), np.concatenate(a["new"])
        row = {k: v for k, v in c.items()}
        row["all"] = ranking_metrics(rec_all, es.targets, es.n_targets)
        row["like"] = ranking_metrics(rec_all, es.like_targets, es.n_like_targets)
        row["new"] = ranking_metrics(rec_new, es.new_targets, es.n_new_targets)
        row["lists"] = list_stats(rec_all, es, info)
        row["lists_new"] = list_stats(rec_new, es, info)
        row["kl"] = float(np.concatenate(a["kl"]).mean())
        for k, v in a.items():
            if k.startswith(("E_", "top10_")):
                row[k] = float(np.concatenate(v).mean())
        results.append(row)
        print(c["name"], f"ndcg@10={row['all']['ndcg@10']:.4f} kl={row['kl']:.3f}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
