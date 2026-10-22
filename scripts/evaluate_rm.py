"""Evaluate reward-model members, their ensembles and non-sequential baselines on held-out events.

Evaluation sets (teacher forcing: each event is predicted from everything before it):
  test     last 24 h of data, all users
  val      validation day, all users
  holdout  users never seen by any reward model, events in [train_end - 14d, train_end - 7d),
           the only set where the 7-day relisten label is observable
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from seqrec_rl import plotting
from seqrec_rl.data import DAY, LAST_TIMESTAMP, N_SPECIAL, TemporalSplit
from seqrec_rl.response_baselines import (aggregate_stats, causal_features, feature_frame, fit_gbm,
                                          mean_predictions, predict_gbm, with_labels)
from seqrec_rl.reward import load_rm, make_event_arrays, predict_targets
from seqrec_rl.reward_data import HEADS, RELISTEN_WINDOW, token_length_buckets
from seqrec_rl.reward_eval import head_metrics, labels_at

COLUMNS = ["uid", "timestamp", "token", "is_organic", "played_ratio_pct", "liked", "disliked", "is_repeat",
           "lp", "cont", "next_lp_ts", "ratio_bucket", "gap_bucket", "flags"]
CONTEXT_KEEP = 520


def metrics_for(labels: dict, probs: np.ndarray, valid_relisten: np.ndarray, mask: np.ndarray | None = None):
    out = {}
    for j, h in enumerate(HEADS):
        m = valid_relisten.copy() if h == "relisten" else np.ones(len(probs), dtype=bool)
        if mask is not None:
            m &= mask
        if m.sum() > 0 and (h != "relisten" or m.any()):
            out[h] = head_metrics(h, labels[h][m], probs[m, j])
    return out


def window_frame(ev: pl.DataFrame, users, upto: int, start: int) -> pl.DataFrame:
    """Events of `users` before `upto`, trimmed to the last CONTEXT_KEEP events before `start`."""
    df = ev.filter(pl.col("uid").is_in(list(users)) & (pl.col("timestamp") < upto))
    # signed ints: first - CONTEXT_KEEP is negative for users with short histories
    df = df.with_columns(pos=pl.int_range(pl.len(), dtype=pl.Int64).over("uid"),
                         first=(pl.col("timestamp") < start).sum().over("uid").cast(pl.Int64))
    df = df.filter(pl.col("pos") >= pl.col("first") - CONTEXT_KEEP)
    # users without events in the window are useless here
    has = df.group_by("uid").agg((pl.col("timestamp") >= start).any().alias("has")).filter("has")
    return df.join(has.select("uid"), on="uid").sort("uid", "timestamp").drop("pos", "first")


def window_targets(ev_arrays, ts: np.ndarray, start: int) -> list[tuple[int, int]]:
    out = []
    for s, e in zip(ev_arrays.offsets[:-1], ev_arrays.offsets[1:]):
        out.append((s + int(np.searchsorted(ts[s:e], start)), int(e)))
    return out


def reliability_plot(y: np.ndarray, preds: dict[str, np.ndarray], title: str, ax, bins: int = 12) -> None:
    for i, (name, p) in enumerate(preds.items()):
        edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
        which = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
        xs = [p[which == b].mean() for b in range(len(edges) - 1) if (which == b).any()]
        ys = [y[which == b].mean() for b in range(len(edges) - 1) if (which == b).any()]
        ax.plot(xs, ys, marker="o", color=plotting.COLORS[i], label=name)
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, lim], [0, lim], color=plotting.MUTED, lw=1, ls="--")
    ax.set_xlabel("предсказанная вероятность")
    ax.set_ylabel("доля в данных")
    ax.set_title(title)
    ax.legend()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed")
    p.add_argument("--rm-dir", default="checkpoints/rm")
    p.add_argument("--members", nargs="+", default=["m0", "m1", "m2", "scratch10"])
    p.add_argument("--proxy", default="m0")
    p.add_argument("--judge", nargs="+", default=["m1", "m2"])
    p.add_argument("--gbm-rows", type=int, default=2_000_000)
    p.add_argument("--out", default="reports/results")
    args = p.parse_args()
    data, out = Path(args.data), Path(args.out)
    split = TemporalSplit.default()
    plotting.setup()

    vocab = pl.read_parquet(data / "vocab.parquet")
    n_tokens = vocab.height + N_SPECIAL
    track_len = np.zeros(n_tokens)
    track_len[vocab["token"].to_numpy()] = vocab["track_length_seconds"].to_numpy()
    len_bucket = token_length_buckets(vocab)

    ev = pl.read_parquet(data / "rm_events.parquet", columns=COLUMNS).sort("uid", "timestamp").with_row_index("row")
    holdout = json.loads((data / "rm_holdout_users.json").read_text())
    t0 = split.train_end - 14 * DAY
    all_users = ev["uid"].unique().to_list()
    sets = {
        "test": dict(users=all_users, upto=LAST_TIMESTAMP + 1, start=split.test_start,
                     label_cutoff=LAST_TIMESTAMP + 1, stats_cutoff=split.train_end),
        "val": dict(users=all_users, upto=split.val_end, start=split.val_start,
                    label_cutoff=LAST_TIMESTAMP + 1, stats_cutoff=split.train_end),
        "holdout": dict(users=holdout, upto=split.train_end - 7 * DAY, start=t0,
                        label_cutoff=split.train_end, stats_cutoff=t0),
    }

    # baseline features over the full stream
    feat = causal_features(with_labels(ev))
    gbm_rows = feat.filter(~pl.col("uid").is_in(holdout) & pl.col("timestamp").is_between(t0, split.train_end, closed="left")
                           & (pl.col("token") >= N_SPECIAL))
    gbm_rows = gbm_rows.sample(min(args.gbm_rows, gbm_rows.height), seed=0).with_columns(
        relisten_valid=pl.col("timestamp") + RELISTEN_WINDOW < split.train_end)
    stats = {c: aggregate_stats(ev, c) for c in {s["stats_cutoff"] for s in sets.values()}}
    gbm = fit_gbm(feature_frame(gbm_rows, *stats[t0], track_len))
    del gbm_rows

    results, slices, disagreement = {}, {}, {}
    for name, s in sets.items():
        df = window_frame(ev, s["users"], s["upto"], s["start"])
        arrays = make_event_arrays(df, s["label_cutoff"])
        targets = window_targets(arrays, df["timestamp"].to_numpy(), s["start"])
        rows = df["row"].to_numpy()

        preds, idx_ref = {}, None
        for m in args.members:
            model, _ = load_rm(Path(args.rm_dir) / m / "best.pt", len_bucket)
            idx, probs = predict_targets(model, arrays, targets, model.config.n_positions)
            assert idx_ref is None or np.array_equal(idx, idx_ref)
            idx_ref, preds[m] = idx, probs
            del model
        gbm_members = [m for m in args.members if m.startswith("m")]
        preds["ensemble"] = np.mean([preds[m] for m in gbm_members], 0)
        preds["judge"] = np.mean([preds[m] for m in args.judge], 0)

        frows = feat.select(pl.all().gather(rows[idx_ref]))
        ff = feature_frame(frows, *stats[s["stats_cutoff"]], track_len)
        preds["gbm"] = predict_gbm(gbm, ff)
        preds.update(mean_predictions(ff, stats[s["stats_cutoff"]][2]))

        labels = labels_at(arrays.cols, idx_ref)
        valid = arrays.cols["relisten_valid"][idx_ref]
        results[name] = {"n_events": int(len(idx_ref)), "n_users": int(len(arrays.uids)),
                         **{k: metrics_for(labels, v, valid) for k, v in preds.items()}}
        print(name, len(idx_ref), {k: round(v["listen_plus"]["roc_auc"], 4) for k, v in results[name].items()
                                   if isinstance(v, dict) and "listen_plus" in v}, flush=True)

        org = arrays.cols["is_organic"][idx_ref] == 1
        rep = arrays.cols["is_repeat"][idx_ref]
        sl = {"organic": org, "recommended": ~org, "repeat": rep, "new": ~rep}
        slices[name] = {sname: {k: metrics_for(labels, preds[k], valid, mask)
                                for k in (args.proxy, "ensemble", "gbm")} for sname, mask in sl.items()}
        stack = np.stack([preds[m] for m in gbm_members])
        disagreement[name] = {sname: dict(zip(HEADS, stack.std(0)[mask].mean(0).tolist()))
                              for sname, mask in {"all": np.ones(len(rep), bool), **sl}.items()}

        if name == "test":
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            sub = {k: preds[k] for k in (args.proxy, "ensemble", "gbm", "item")}
            reliability_plot(labels["listen_plus"], {k: v[:, 1] for k, v in sub.items()}, "Listen+, test", axes[0])
            reliability_plot(labels["like"], {k: v[:, 2] for k, v in sub.items()}, "лайк, test", axes[1])
            plotting.save(fig, Path("reports/figures/rm_calibration.png"))

    out.mkdir(parents=True, exist_ok=True)
    (out / "rm_eval.json").write_text(json.dumps({"results": results, "slices": slices,
                                                  "disagreement": disagreement}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
