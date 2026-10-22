"""Five-step generation from the test states: generated vs real continuation of the history.

For each user the policy generates tracks a_1..a_5, appending each to its own
context (and to the reward model's, with an assumed response). In parallel the
same policy is run on the real continuation: the user's first five listens of the
test day with their real responses. At every step we compare the two:
relevance of the top-10 to the user's remaining Listen+ tracks, KL between the
two next-track distributions, and the reward of the generated track according to
the proxy and the judges under each assumed response.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
import torch

from seqrec_rl.data import LISTEN_PLUS_PCT, N_SPECIAL, TemporalSplit
from seqrec_rl.evaluation import ItemInfo, build_eval_set, ranking_metrics
from seqrec_rl.model import load_model
from seqrec_rl.reward import history_mask, load_rm, make_event_arrays
from seqrec_rl.reward_data import HEADS, token_length_buckets
from seqrec_rl.rl import NEG, cutoff_states, kl, pi_star_logp, policy_logits
from seqrec_rl.rollout import FEEDBACK_MODES, RMSeq, item_probs

RATIO, LIKE = HEADS.index("ratio"), HEADS.index("like")


def test_futures(data: Path, lis: pl.DataFrame, split: TemporalSplit, uids: np.ndarray, steps: int):
    """Real test-day events per user: first `steps` with reward-model features, and all for targets."""
    t = split.test(lis).filter(pl.col("uid").is_in(uids.tolist())).sort("uid", "timestamp")
    ev = (pl.read_parquet(data / "rm_events.parquet")
          .filter(pl.col("uid").is_in(uids.tolist()) & (pl.col("timestamp") >= split.test_start))
          .sort("uid", "timestamp"))
    assert np.array_equal(t["token"].to_numpy(), ev["token"].to_numpy())
    arr = make_event_arrays(ev, 0)
    per_user = t.group_by("uid", maintain_order=True).agg("item_id", "token", "played_ratio_pct")
    fut = {}
    for (uid, items, toks, ratio), s in zip(per_user.iter_rows(), arr.offsets[:-1]):
        fut[int(uid)] = {
            "items": np.array(items), "tokens": np.array(toks), "ratio": np.array(ratio),
            "feat": {c: arr.cols[c][s:s + steps].astype(np.int64)
                     for c in ("token", "ratio_bucket", "flags", "gap_bucket")},
        }
    return fut


def targets_after(f: dict, k: int) -> tuple[np.ndarray, int]:
    keep = f["ratio"][k:] >= LISTEN_PLUS_PCT
    items, toks = f["items"][k:][keep], f["tokens"][k:][keep]
    return np.unique(toks[toks >= N_SPECIAL]), len(np.unique(items))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed")
    p.add_argument("--base", default="checkpoints/base/best.pt")
    p.add_argument("--rl", default="checkpoints/rl/ratio_b0.3_exact/policy.pt")
    p.add_argument("--positive", default="checkpoints/rl/positive_ce/policy.pt")
    p.add_argument("--rm-dir", default="checkpoints/rm")
    p.add_argument("--scales", default="runs/rl/reward_scales.json")
    p.add_argument("--users", type=int, default=2000)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--out", default="reports/results/rollouts.json")
    args = p.parse_args()
    data, K = Path(args.data), args.steps
    split = TemporalSplit.default()
    torch.manual_seed(0)

    lis = pl.read_parquet(data / "listens.parquet")
    likes = pl.read_parquet(data / "likes.parquet")
    vocab = pl.read_parquet(data / "vocab.parquet")
    info = ItemInfo.from_vocab(vocab)
    n_tokens = len(info.train_count)
    es = build_eval_set(lis, likes, split, "test")
    n_test = split.test(lis).group_by("uid").len()
    enough = set(n_test.filter(pl.col("len") > K)["uid"].to_list())
    cand = np.array([i for i, u in enumerate(es.uids) if int(u) in enough])
    sel = np.sort(np.random.default_rng(0).choice(cand, min(args.users, len(cand)), replace=False))
    uids = es.uids[sel]
    fut = test_futures(data, lis, split, uids, K)
    del lis
    arrays, row_of = cutoff_states(data / "rm_events.parquet", uids, split.val_end)

    lb = token_length_buckets(vocab)
    rms = {m: load_rm(Path(args.rm_dir) / m / "best.pt", lb)[0] for m in ("m0", "m1", "m2")}
    scale = json.loads(Path(args.scales).read_text())["ratio"]
    base = load_model(args.base)[0].eval()
    rl = load_model(args.rl)[0].eval()
    pos = load_model(args.positive)[0].eval()
    policies = {
        "GPT-2": dict(model=base),
        "π* ratio β=1": dict(model=base, beta=1.0, mode="unknown"),
        "π* ratio β=0.3": dict(model=base, beta=0.3, mode="unknown"),
        "π* ratio β=0.3, оптимистичный отклик": dict(model=base, beta=0.3, mode="optimistic"),
        "RL ratio β=0.3": dict(model=rl),
        "Positive-CE": dict(model=pos),
    }
    runs = [(n, "greedy") for n in policies] + [("GPT-2", "sample"), ("π* ratio β=0.3", "sample")]

    def decide(spec, pol_ctx, rm_seq: RMSeq, hist):
        with torch.no_grad():
            logp = torch.log_softmax(policy_logits(spec["model"], pol_ctx), 1)
            if "beta" in spec:
                h = rm_seq.encode(rms["m0"])
                r = torch.sigmoid(rms["m0"].score_all(h, hist, ("ratio",))["ratio"])
                logp = pi_star_logp(logp, r / scale, spec["beta"])
        return logp

    acc = {f"{n}|{d}": {"lists_gen": [[] for _ in range(K + 1)], "lists_real": [[] for _ in range(K + 1)],
                        "kl_gen_real": [[] for _ in range(K + 1)], "items": [], "loop": [],
                        "reward": {m: {w: [] for w in ("proxy", "judge", "std", "like_proxy", "like_judge")}
                                   for m in FEEDBACK_MODES},
                        "top5_reward": {w: [] for w in ("proxy", "judge")}} for n, d in runs}
    real_eval = {"pred_proxy": [], "pred_judge": [], "actual": []}
    tk = [[targets_after(fut[int(u)], k) for u in uids] for k in range(K + 1)]

    for start in range(0, len(uids), args.batch):
        idx = np.arange(start, min(start + args.batch, len(uids)))
        bu = uids[idx]
        rows = np.array([row_of[int(u)] for u in bu])
        rm0 = RMSeq.from_events(arrays, arrays.offsets[rows], arrays.offsets[rows + 1])
        hist0 = [es.history[sel[i]] for i in idx]
        pol0 = [es.hist_tokens[sel[i]][-512:].astype(np.int64) for i in idx]
        real_feat = [fut[int(u)]["feat"] for u in bu]
        real_tok = np.stack([f["token"] for f in real_feat])

        for name, dec in runs:
            spec, a = policies[name], acc[f"{name}|{dec}"]
            dec_mode = spec.get("mode", "unknown")
            pol_g, pol_r = [x.copy() for x in pol0], [x.copy() for x in pol0]
            rm_g = {m: rm0.copy() for m in FEEDBACK_MODES}
            rm_r = rm0.copy()
            hist_g, hist_r = history_mask(hist0, n_tokens), history_mask(hist0, n_tokens)
            generated = torch.zeros_like(hist_g)
            steps_items = []
            for k in range(K + 1):
                lg = decide(spec, pol_g, rm_g[dec_mode], hist_g)
                lr = decide(spec, pol_r, rm_r, hist_r)
                a["lists_gen"][k].append(lg.topk(10, 1).indices.cpu().numpy())
                a["lists_real"][k].append(lr.topk(10, 1).indices.cpu().numpy())
                a["kl_gen_real"][k].append(kl(lg, lr).cpu().numpy())
                if k == 0:
                    top5 = lg.topk(5, 1).indices
                    with torch.no_grad():
                        hs = {m: rm_g["unknown"].encode(rms[m]) for m in rms}
                        r5 = {m: torch.stack([item_probs(rms[m], hs[m], top5[:, j], hist_g.gather(1, top5[:, j:j + 1])[:, 0])[:, RATIO]
                                              for j in range(5)], 1) for m in rms}
                    a["top5_reward"]["proxy"].append(r5["m0"].mean(1).cpu().numpy())
                    a["top5_reward"]["judge"].append(((r5["m1"] + r5["m2"]) / 2).mean(1).cpu().numpy())
                if k == K:
                    break
                with torch.no_grad():
                    a["loop"].append((generated.gather(1, lg.argmax(1, keepdim=True))[:, 0]).cpu().numpy())
                    if dec == "greedy":
                        act = lg.masked_fill(generated, NEG).argmax(1)
                    else:
                        act = torch.multinomial(lg.exp(), 1)[:, 0]
                    rep = hist_g.gather(1, act[:, None])[:, 0]
                    pred_expected = None
                    for m in FEEDBACK_MODES:
                        pr = {name_: item_probs(rms[name_], rm_g[m].encode(rms[name_]), act, rep) for name_ in rms}
                        stack = torch.stack([pr["m0"], pr["m1"], pr["m2"]])
                        rw = a["reward"][m]
                        rw["proxy"].append(pr["m0"][:, RATIO].cpu().numpy())
                        rw["judge"].append(((pr["m1"] + pr["m2"]) / 2)[:, RATIO].cpu().numpy())
                        rw["std"].append(stack[:, :, RATIO].std(0).cpu().numpy())
                        rw["like_proxy"].append(pr["m0"][:, LIKE].cpu().numpy())
                        rw["like_judge"].append(((pr["m1"] + pr["m2"]) / 2)[:, LIKE].cpu().numpy())
                        if m == "expected":
                            pred_expected = pr["m0"][:, RATIO].cpu().numpy()
                    if name == "GPT-2" and dec == "greedy":
                        e_k = torch.as_tensor(real_tok[:, k], device="cuda")
                        ok = e_k >= N_SPECIAL
                        e_k = torch.where(ok, e_k, torch.full_like(e_k, N_SPECIAL))
                        rep_r = hist_r.gather(1, e_k[:, None])[:, 0]
                        pr = {name_: item_probs(rms[name_], rm_r.encode(rms[name_]), e_k, rep_r)[:, RATIO] for name_ in rms}
                        okn = ok.cpu().numpy()
                        real_eval["pred_proxy"].append(pr["m0"].cpu().numpy()[okn])
                        real_eval["pred_judge"].append(((pr["m1"] + pr["m2"]) / 2).cpu().numpy()[okn])
                        real_eval["actual"].append(np.array([min(fut[int(u)]["ratio"][k], 100) / 100 for u in bu])[okn])
                act_np = act.cpu().numpy()
                steps_items.append(act_np)
                pol_g =[np.append(x, t) for x, t in zip(pol_g, act_np)]
                for m in FEEDBACK_MODES:
                    rm_g[m].generated(act_np, m, pred_expected)
                hist_g[torch.arange(len(act)), act] = True
                generated[torch.arange(len(act)), act] = True
                pol_r = [np.append(x, f["token"][k]) for x, f in zip(pol_r, real_feat)]
                rm_r.append(**{c: np.array([f[c][k] for f in real_feat]) for c in ("token", "ratio_bucket", "flags",
                                                                                   "gap_bucket")})
                ok_tok = real_tok[:, k] >= N_SPECIAL
                hist_r[torch.as_tensor(np.flatnonzero(ok_tok), device="cuda"),
                       torch.as_tensor(real_tok[ok_tok, k], device="cuda")] = True
            a["items"].append(np.stack(steps_items, 1))
        print(f"{start + len(idx)}/{len(uids)}", flush=True)

    hits_t0 = tk[0]
    results = {}
    for key, a in acc.items():
        items = np.concatenate(a["items"])  # (n, K)
        res = {"steps": []}
        for k in range(K + 1):
            tg = [t[0] for t in tk[k]]
            nt = np.array([t[1] for t in tk[k]])
            gen, real = np.concatenate(a["lists_gen"][k]), np.concatenate(a["lists_real"][k])
            row = {"k": k, "gen": ranking_metrics(gen, tg, nt, ks=(10,)), "real": ranking_metrics(real, tg, nt, ks=(10,)),
                   "kl_gen_real": float(np.concatenate(a["kl_gen_real"][k]).mean()),
                   "jaccard_gen_real": float(np.mean([len(np.intersect1d(g, r)) / len(np.union1d(g, r))
                                                      for g, r in zip(gen, real)]))}
            if k < K:
                it = items[:, k]
                prev = items[:, k - 1] if k > 0 else None
                row |= {
                    "item_hit_test_listen_plus": float(np.mean([t in tg0[0] for t, tg0 in zip(it, hits_t0)])),
                    "item_in_history": float(np.mean([np.isin(t, es.history[sel[i]]) for i, t in enumerate(it)])),
                    "item_length_sec": float(info.track_length[it].mean()),
                    "item_log10_pop": float(np.log10(info.train_count[it]).mean()),
                    "same_artist_as_prev": None if prev is None else float((info.artist[it] == info.artist[prev]).mean()),
                    "loop_attempt": float(np.concatenate(a["loop"][k::K]).mean()) if a["loop"] else None,
                    "reward": {m: {w: float(np.concatenate(v[k::K]).mean()) for w, v in a["reward"][m].items()}
                               for m in FEEDBACK_MODES},
                }
            res["steps"].append(row)
        slate_targets = [t[0] for t in tk[0]]
        nt0 = np.array([t[1] for t in tk[0]])
        topk = np.concatenate(a["lists_gen"][0])[:, :5]
        res["slate"] = {
            "next_k": ranking_metrics(items, slate_targets, nt0, ks=(5,)),
            "top_k": ranking_metrics(topk, slate_targets, nt0, ks=(5,)),
            "next_k_artists": float(np.mean([len(np.unique(info.artist[r])) for r in items])),
            "top_k_artists": float(np.mean([len(np.unique(info.artist[r])) for r in topk])),
            "next_k_reward": {m: {w: float(np.concatenate(v).mean()) for w, v in a["reward"][m].items()}
                              for m in FEEDBACK_MODES},
            "top_k_reward": {w: float(np.concatenate(v).mean()) for w, v in a["top5_reward"].items()},
        }
        results[key] = res
        print(key, "next-K ndcg@5", round(res["slate"]["next_k"]["ndcg@5"], 4),
              "top-K ndcg@5", round(res["slate"]["top_k"]["ndcg@5"], 4), flush=True)
    actual = np.concatenate(real_eval["actual"])
    results["real_items"] = {
        "n": int(len(actual)), "actual_mean": float(actual.mean()),
        "proxy_mean": float(np.concatenate(real_eval["pred_proxy"]).mean()),
        "judge_mean": float(np.concatenate(real_eval["pred_judge"]).mean()),
        "proxy_mae": float(np.abs(np.concatenate(real_eval["pred_proxy"]) - actual).mean()),
    }
    out = Path(args.out)
    out.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
