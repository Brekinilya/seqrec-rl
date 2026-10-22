"""Fine-tune the GPT-2 policy against a reward-model reward with a KL penalty to the base model.

--method exact    gradient of E_pi[r] - beta KL computed over the whole vocabulary
--method sampled  REINFORCE with a leave-one-out baseline over `group` sampled tracks per state
                  (the estimator used in language-model RLHF, same objective) plus the exact KL

The log tracks the objective of the policy and of the closed-form optimum pi* on
the same batch, so the gap shows how far training is from the best possible policy.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

from seqrec_rl.data import N_SPECIAL, TemporalSplit
from seqrec_rl.model import load_model, save_model
from seqrec_rl.reward import history_mask, load_rm
from seqrec_rl.reward_data import token_length_buckets
from seqrec_rl.rl import (TrainStates, encode_windows, exact_loss, expectation, kl, pi_star_logp, policy_logits,
                          rm_rewards, sampled_loss)

from rl_scales import train_event_arrays  # noqa: E402  (same scripts folder)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed")
    p.add_argument("--base", default="checkpoints/base/best.pt")
    p.add_argument("--rm", default="checkpoints/rm/m0/best.pt")
    p.add_argument("--scales", default="runs/rl/reward_scales.json")
    p.add_argument("--reward", default="ratio")
    p.add_argument("--beta", type=float, default=0.3, help="KL weight, rewards in units of their scale")
    p.add_argument("--method", choices=["exact", "sampled"], default="exact")
    p.add_argument("--group", type=int, default=16)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--users", type=int, default=0, help="subsample users for training states (0 = all)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))
    data = Path(args.data)
    split = TemporalSplit.default()
    vocab = pl.read_parquet(data / "vocab.parquet")
    track_len = np.zeros(vocab.height + N_SPECIAL, dtype=np.float32)
    track_len[vocab["token"].to_numpy()] = vocab["track_length_seconds"].to_numpy()
    track_len = torch.as_tensor(track_len, device="cuda")
    scale = json.loads(Path(args.scales).read_text())[args.reward]

    ev = train_event_arrays(data, split, args.users or None, seed=args.seed)
    states = TrainStates(ev, 512, seed=args.seed)
    policy, _ = load_model(args.base)
    ref, _ = load_model(args.base)
    rm, _ = load_rm(args.rm, token_length_buckets(vocab))
    policy.eval()  # no dropout: the policy distribution must be the one being optimised
    ref.eval()
    for prm in ref.parameters():
        prm.requires_grad_(False)
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=0.0, betas=(0.9, 0.98), fused=True)
    n_tokens = policy.config.vocab_size

    log = open(out / "log.jsonl", "w", encoding="utf-8")
    t0 = time.time()
    run = {k: 0.0 for k in ("loss", "er", "kl", "j", "j_star", "er_star", "kl_star")}
    for step in range(1, args.steps + 1):
        contexts, history, windows = states.sample(args.batch)
        with torch.no_grad():
            ref_logp = torch.log_softmax(policy_logits(ref, contexts), 1)
            hist = history_mask(history, n_tokens)
            r = rm_rewards(rm, encode_windows(rm, ev, windows), hist, track_len, [args.reward])[args.reward] / scale
        frac = step / args.steps
        f = min(1.0, step / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
        for g in opt.param_groups:
            g["lr"] = args.lr * f
        logits = policy_logits(policy, contexts)
        if args.method == "exact":
            loss, er, k = exact_loss(logits, ref_logp, r, args.beta)
        else:
            loss, er, k = sampled_loss(logits, ref_logp, r, args.beta, args.group)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

        with torch.no_grad():
            star = pi_star_logp(ref_logp, r, args.beta)
            er_s, k_s = expectation(star, r), kl(star, ref_logp)
            base_er = expectation(ref_logp, r)
        for key, v in (("loss", loss.item()), ("er", (er - base_er).mean().item()), ("kl", k.mean().item()),
                       ("j", (er - base_er - args.beta * k).mean().item()),
                       ("er_star", (er_s - base_er).mean().item()), ("kl_star", k_s.mean().item()),
                       ("j_star", (er_s - base_er - args.beta * k_s).mean().item())):
            run[key] += v
        if step % 50 == 0:
            rec = {"step": step, **{k: v / 50 for k, v in run.items()}, "elapsed_min": (time.time() - t0) / 60}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            run = {k: 0.0 for k in run}
    save_model(policy, out / "policy.pt", args=vars(args), step=args.steps)
    log.close()


if __name__ == "__main__":
    main()
