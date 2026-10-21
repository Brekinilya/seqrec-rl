"""Train the GPT-2 next-item model on the train part of the temporal split."""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

from seqrec_rl.data import TemporalSplit, user_sequences
from seqrec_rl.evaluation import ItemInfo, build_eval_set, evaluate
from seqrec_rl.model import GPTRecommender, WindowSampler, build_model, next_item_loss, save_model


def val_summary(res: dict) -> dict:
    return {
        "ndcg@10": res["all"]["listen_plus"]["ndcg@10"],
        "recall@100": res["all"]["listen_plus"]["recall@100"],
        "new_ndcg@10": res["new"]["listen_plus"]["ndcg@10"],
        "new_recall@100": res["new"]["listen_plus"]["recall@100"],
        "in_history@10": res["all"]["lists"]["in_history@10"],
        "coverage@100": res["all"]["lists"]["coverage@100"],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed")
    p.add_argument("--out", default="checkpoints/base")
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--evals-per-epoch", type=int, default=2)
    p.add_argument("--chunk", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=0, help="stop early (smoke tests)")
    p.add_argument("--resume", default="", help="checkpoint to continue from (same args)")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "log.jsonl", "a", encoding="utf-8")

    data = Path(args.data)
    lis = pl.read_parquet(data / "listens.parquet")
    likes = pl.read_parquet(data / "likes.parquet")
    info = ItemInfo.from_vocab(pl.read_parquet(data / "vocab.parquet"))
    split = TemporalSplit.default()

    _, offsets, tokens = user_sequences(split.train(lis).sort("uid", "timestamp"))
    sampler = WindowSampler(offsets, tokens, args.seq_len, seed=args.seed)
    val = build_eval_set(lis, likes, split, "val")
    del lis

    model = build_model(len(info.train_count), args.seq_len, args.d_model, args.layers, args.heads,
                        args.dropout).cuda()
    decay = [p_ for n, p_ in model.named_parameters() if p_.dim() >= 2]
    no_decay = [p_ for n, p_ in model.named_parameters() if p_.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.98), fused=True)
    steps_per_epoch = sampler.n_batches(args.batch)
    total = args.epochs * steps_per_epoch
    eval_every = steps_per_epoch // args.evals_per_epoch

    step, best, rewarm_from = 0, -1.0, None
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state"])
        step = ckpt["step"]
        if "optim" in ckpt:
            opt.load_state_dict(ckpt["optim"])
        else:
            rewarm_from = step  # fresh Adam moments: short warmup again
        if (out / "best.pt").exists():
            best = torch.load(out / "best.pt", map_location="cpu", weights_only=False)["val"]["ndcg@10"]
        log.close()
        # forget log records written by the interrupted run after the checkpoint
        lines = (out / "log.jsonl").read_text(encoding="utf-8").splitlines()
        kept = [line for line in lines if json.loads(line).get("step", 0) <= step]
        (out / "log.jsonl").write_text("".join(line + "\n" for line in kept), encoding="utf-8")
        log = open(out / "log.jsonl", "a", encoding="utf-8")
        log.write(json.dumps({"resume_from": step}) + "\n")
    else:
        (out / "args.json").write_text(json.dumps(vars(args), indent=2))

    def lr_at(s):
        if s < args.warmup:
            f = s / args.warmup
        else:
            t = min(1.0, (s - args.warmup) / max(1, total - args.warmup))
            f = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t))
        if rewarm_from is not None:
            f *= min(1.0, (s - rewarm_from + 1) / 200)
        return args.lr * f

    print(f"params {sum(p_.numel() for p_ in model.parameters()) / 1e6:.1f}M, "
          f"{steps_per_epoch} steps/epoch, {total} total, starting at {step}", flush=True)

    t0 = time.time()
    run_loss, run_n, run_tok = 0.0, 0, 0
    epoch = round(step / steps_per_epoch)
    done = False
    while not done:
        sampler.rng = np.random.default_rng(args.seed + epoch)
        for batch in sampler.batches(args.batch):
            model.train()
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            x = torch.from_numpy(batch).cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, n = next_item_loss(model, x[:, :-1], x[:, 1:], args.chunk)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            run_loss += loss.item() * n
            run_n += n
            run_tok += n

            if step % 200 == 0:
                rec = {"step": step, "epoch": epoch, "loss": run_loss / run_n, "lr": lr_at(step),
                       "tok_per_s": run_tok / (time.time() - t0), "elapsed_min": (time.time() - t0) / 60}
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
                run_loss, run_n = 0.0, 0

            done = step >= total or bool(args.max_steps and step >= args.max_steps)
            if step % eval_every == 0 or done:
                res, _ = evaluate(GPTRecommender(model), val, info)
                summary = {"step": step, "epoch": epoch, "val": val_summary(res)}
                print(json.dumps(summary), flush=True)
                log.write(json.dumps(summary) + "\n")
                log.flush()
                save_model(model, out / "last.pt", step=step, val=summary["val"], args=vars(args),
                           optim=opt.state_dict())
                if summary["val"]["ndcg@10"] > best:
                    best = summary["val"]["ndcg@10"]
                    save_model(model, out / "best.pt", step=step, val=summary["val"], args=vars(args))
            if done:
                break
        epoch += 1
    log.close()


if __name__ == "__main__":
    main()
