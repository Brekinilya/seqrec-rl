"""Filtered behaviour cloning: continue next-item training of the base model on Listen+ targets only.

A supervised alternative to RL for the played-ratio reward: no reward model, the
policy simply stops imitating listens that were skipped.
"""
import argparse
import json
import math
import time
from pathlib import Path

import polars as pl
import torch

from seqrec_rl.data import LISTEN_PLUS_PCT, PAD, TemporalSplit, user_sequences
from seqrec_rl.model import WindowSampler, load_model, next_item_loss, save_model


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed")
    p.add_argument("--base", default="checkpoints/base/best.pt")
    p.add_argument("--out", default="checkpoints/rl/positive_ce")
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))

    split = TemporalSplit.default()
    train = split.train(pl.read_parquet(Path(args.data) / "listens.parquet")).sort("uid", "timestamp")
    _, offsets, tokens = user_sequences(train)
    ratio = train["played_ratio_pct"].to_numpy()
    sampler = WindowSampler(offsets, tokens, 512, seed=args.seed + 7)
    model, _ = load_model(args.base)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98), fused=True)

    log = open(out / "log.jsonl", "w", encoding="utf-8")
    t0, step, run, run_n = time.time(), 0, 0.0, 0
    for batch, r in sampler.batches(args.batch, aux=ratio):
        model.train()
        f = min(1.0, (step + 1) / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / args.steps)))
        for g in opt.param_groups:
            g["lr"] = args.lr * f
        x = torch.from_numpy(batch).cuda()
        labels = x[:, 1:].clone()
        labels[torch.from_numpy(r[:, 1:]).cuda() < LISTEN_PLUS_PCT] = PAD
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, n = next_item_loss(model, x[:, :-1], labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        run += loss.item()
        run_n += 1
        if step % 100 == 0:
            rec = {"step": step, "loss": run / run_n, "elapsed_min": (time.time() - t0) / 60}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            run, run_n = 0.0, 0
        if step >= args.steps:
            break
    save_model(model, out / "policy.pt", args=vars(args), step=step)
    log.close()


if __name__ == "__main__":
    main()
