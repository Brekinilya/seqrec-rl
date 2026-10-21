"""Train one reward-model ensemble member on a random subset of the non-holdout users."""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

from seqrec_rl.data import TemporalSplit
from seqrec_rl.model import WindowSampler
from seqrec_rl.reward import RewardModel, make_event_arrays, predict_targets, rm_loss, save_rm
from seqrec_rl.reward_data import token_length_buckets
from seqrec_rl.reward_eval import all_metrics, labels_at

SHORT_HEADS = ("ratio", "listen_plus", "like", "dislike", "cont")


def window_targets(ev, ts: np.ndarray, start: int, end: int) -> list[tuple[int, int]]:
    """Per user: global index range of events with start <= timestamp < end."""
    out = []
    for s, e in zip(ev.offsets[:-1], ev.offsets[1:]):
        t = ts[s:e]
        out.append((s + int(np.searchsorted(t, start)), s + int(np.searchsorted(t, end))))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed")
    p.add_argument("--init", default="checkpoints/base/best.pt")
    p.add_argument("--scratch", action="store_true", help="random init instead of --init")
    p.add_argument("--out", default="checkpoints/rm/m0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--user-frac", type=float, default=0.8)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--evals-per-epoch", type=int, default=2)
    p.add_argument("--feedback-dropout", type=float, default=0.05)
    p.add_argument("--val-users", type=int, default=1500)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--resume", action="store_true", help="continue from <out>/last.pt with the same args")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        (out / "args.json").write_text(json.dumps(vars(args), indent=2))

    data = Path(args.data)
    split = TemporalSplit.default()
    vocab = pl.read_parquet(data / "vocab.parquet")
    len_bucket = token_length_buckets(vocab)
    ev_df = pl.read_parquet(data / "rm_events.parquet")
    holdout = set(json.loads((data / "rm_holdout_users.json").read_text()))
    pool = np.array(sorted(set(ev_df["uid"].unique().to_list()) - holdout))
    members = np.sort(rng.choice(pool, int(len(pool) * args.user_frac), replace=False))
    (out / "users.json").write_text(json.dumps(members.tolist()))

    train_ev = make_event_arrays(
        ev_df.filter(pl.col("uid").is_in(members.tolist()) & (pl.col("timestamp") < split.train_end))
        .sort("uid", "timestamp"), label_cutoff=split.train_end)
    val_users = np.sort(rng.choice(pool, args.val_users, replace=False))
    val_df = (ev_df.filter(pl.col("uid").is_in(val_users.tolist()) & (pl.col("timestamp") < split.val_end))
              .sort("uid", "timestamp"))
    val_ev = make_event_arrays(val_df, label_cutoff=split.train_end)
    val_targets = window_targets(val_ev, val_df["timestamp"].to_numpy(), split.val_start, split.val_end)
    del ev_df, val_df

    model = RewardModel(len(len_bucket), len_bucket, args.seq_len).cuda()
    if not args.scratch:
        model.init_from_policy(torch.load(args.init, map_location="cpu", weights_only=False)["state"])
    decay = [p_ for p_ in model.parameters() if p_.dim() >= 2]
    no_decay = [p_ for p_ in model.parameters() if p_.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.98), fused=True)

    sampler = WindowSampler(train_ev.offsets, train_ev.cols["token"], args.seq_len, seed=args.seed)
    steps_per_epoch = sampler.n_batches(args.batch)
    total = args.epochs * steps_per_epoch
    eval_every = steps_per_epoch // args.evals_per_epoch

    step, best, rewarm_from = 0, math.inf, None
    if args.resume:
        ckpt = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state"])
        step = ckpt["step"]
        if "optim" in ckpt:
            opt.load_state_dict(ckpt["optim"])
        else:
            rewarm_from = step  # fresh Adam moments: short warmup again
        best = torch.load(out / "best.pt", map_location="cpu", weights_only=False)["val_bce_sum"]
        lines = (out / "log.jsonl").read_text(encoding="utf-8").splitlines()
        kept = [line for line in lines if json.loads(line).get("step", 0) <= step]
        (out / "log.jsonl").write_text("".join(line + "\n" for line in kept), encoding="utf-8")
    log = open(out / "log.jsonl", "a" if args.resume else "w", encoding="utf-8")

    def lr_at(s):
        if s < args.warmup:
            f = s / args.warmup
        else:
            t = min(1.0, (s - args.warmup) / max(1, total - args.warmup))
            f = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t))
        if rewarm_from is not None:
            f *= min(1.0, (s - rewarm_from + 1) / 100)
        return args.lr * f

    print(f"members {len(members)}, train events {len(train_ev.cols['token'])}, "
          f"{steps_per_epoch} steps/epoch, {total} total", flush=True)

    def validate(step):
        idx, probs = predict_targets(model, val_ev, val_targets, args.seq_len)
        m = all_metrics(labels_at(val_ev.cols, idx), probs, heads=SHORT_HEADS)
        score = sum(m[h]["bce"] for h in SHORT_HEADS)
        rec = {"step": step, "val_bce_sum": score,
               "val": {h: {k: v for k, v in r.items() if k in ("bce", "mae", "roc_auc", "pr_auc")}
                       for h, r in m.items()}}
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        return score

    t0 = time.time()
    run, run_n = torch.zeros(6), 0
    done = step >= total
    # every epoch's window order depends only on its seed, so a resumed run skips exactly the seen batches
    epoch, skip = divmod(step, steps_per_epoch)
    while not done:
        sampler.rng = np.random.default_rng(args.seed * 100 + epoch)
        w = sampler.windows()
        for i in range(skip * args.batch, len(w), args.batch):
            model.train()
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            batch = train_ev.gather(w[i:i + args.batch], args.seq_len + 1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, per_head = rm_loss(model, batch, args.feedback_dropout)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            run += per_head.cpu()
            run_n += 1
            if step % 200 == 0:
                rec = {"step": step, "epoch": epoch, "loss": dict(zip(
                    ("ratio", "listen_plus", "like", "dislike", "cont", "relisten"), (run / run_n).tolist())),
                       "lr": lr_at(step), "elapsed_min": (time.time() - t0) / 60}
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
                run, run_n = torch.zeros(6), 0
            done = step >= total or bool(args.max_steps and step >= args.max_steps)
            if step % eval_every == 0 or done:
                score = validate(step)
                save_rm(model, out / "last.pt", step=step, args=vars(args), optim=opt.state_dict())
                if score < best:
                    best = score
                    save_rm(model, out / "best.pt", step=step, args=vars(args), val_bce_sum=score)
            if done:
                break
        epoch, skip = epoch + 1, 0
    log.close()
    (out / "done").write_text(str(step))


if __name__ == "__main__":
    main()
