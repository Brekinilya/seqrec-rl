"""Figures and tables for stage 3 from reports/results/rl_sweep.json (and RL runs, if evaluated)."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from seqrec_rl import plotting

REWARD_ORDER = ["ratio", "like", "like_plus_ratio", "seconds", "cont", "relisten", "discovery"]
TITLES = {"ratio": "доля дослушивания", "like": "P(лайк)", "like_plus_ratio": "лайк + 0.1·доля",
          "seconds": "секунды прослушивания", "cont": "продолжение сессии", "relisten": "возврат к треку",
          "discovery": "новый трек + возврат"}


def load(path: Path) -> tuple[dict, list]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    base = next(r for r in rows if r["name"] == "base")
    return base, rows


def curve(rows, reward, defence="none", kappa=None):
    sel = [r for r in rows if r.get("reward") == reward and r.get("defence") == defence
           and (kappa is None or r.get("kappa") == kappa)]
    return sorted(sel, key=lambda r: -r["beta"])


def gain(row, base, who, reward, kind="E"):
    return row[f"{kind}_{who}_{reward}"] - base[f"{kind}_{who}_{reward}"]


def plot_pareto(base, rows, rl_rows, out):
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    for ax, reward in zip(axes.ravel(), REWARD_ORDER):
        c = curve(rows, reward)
        for who, color, ls in (("judge", plotting.COLORS[0], "-"), ("proxy", plotting.COLORS[1], "--")):
            xs = [0.0] + [gain(r, base, who, reward) for r in c]
            ys = [base["all"]["ndcg@10"]] + [r["all"]["ndcg@10"] for r in c]
            ax.plot(xs, ys, marker="o", ms=3, color=color, ls=ls, label=f"π*, награда по {'судьям' if who == 'judge' else 'proxy'}")
        for i, r in enumerate(x for x in rl_rows if x["reward"] == reward):
            ax.scatter([gain(r, base, "judge", reward)], [r["all"]["ndcg@10"]], s=60, marker="*",
                       color=plotting.COLORS[2 + i % 4], zorder=5, label=r["label"])
        ax.set_title(TITLES[reward])
        ax.set_xlabel("прирост E[r] к базовой модели")
        ax.set_ylabel("NDCG@10")
        ax.legend(fontsize=6)
    axes.ravel()[-1].axis("off")
    plotting.save(fig, out)


def plot_overoptimisation(base, rows, out):
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4))
    for ax, reward in zip(axes, ["ratio", "like", "like_plus_ratio", "relisten"]):
        c = curve(rows, reward)
        x = [0.0] + [np.sqrt(r["kl"]) for r in c]
        for who, color, ls in (("proxy", plotting.COLORS[1], "--"), ("judge", plotting.COLORS[0], "-")):
            ax.plot(x, [0.0] + [gain(r, base, who, reward) for r in c], marker="o", ms=3, color=color, ls=ls,
                    label="proxy (m0)" if who == "proxy" else "судьи (m1, m2)")
        ax.set_title(TITLES[reward])
        ax.set_xlabel("√KL(π ‖ π_ref)")
        ax.set_ylabel("прирост E[r]")
        ax.legend()
    plotting.save(fig, out)


def plot_behaviour(base, rows, out):
    panels = [("track_length_sec@10", "длина трека в top-10, с"), ("in_history@10", "доля треков из истории"),
              ("log10_train_count@10", "log10 популярности"), ("unique_artists@10", "артистов в top-10"),
              ("coverage@100", "coverage@100")]
    show = ["ratio", "like", "like_plus_ratio", "cont", "relisten", "discovery"]
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.4))
    for ax, (key, title) in zip(axes, panels):
        for i, reward in enumerate(show):
            c = curve(rows, reward)
            ax.plot([0.0] + [np.sqrt(r["kl"]) for r in c], [base["lists"][key]] + [r["lists"][key] for r in c],
                    marker="o", ms=3, color=plotting.COLORS[i], label=TITLES[reward])
        ax.set_title(title)
        ax.set_xlabel("√KL(π ‖ π_ref)")
    axes[0].legend(fontsize=7)
    plotting.save(fig, out)


def plot_defences(base, rows, out):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    variants = [("none", None, "только KL"), ("pess", 1, "пессимизм κ=1"), ("pess", 3, "пессимизм κ=3"),
                ("top200", None, "кандидаты: top-200 π_ref")]
    for ax, reward in zip(axes, ["ratio", "like", "like_plus_ratio"]):
        for i, (d, k, label) in enumerate(variants):
            c = curve(rows, reward, d, k)
            ax.plot([0.0] + [gain(r, base, "judge", reward) for r in c],
                    [base["all"]["ndcg@10"]] + [r["all"]["ndcg@10"] for r in c],
                    marker="o", ms=3, color=plotting.COLORS[i], label=label)
        ax.set_title(f"{TITLES[reward]}: защиты")
        ax.set_xlabel("прирост E[r] по судьям")
        ax.set_ylabel("NDCG@10")
        ax.legend(fontsize=7)
    plotting.save(fig, out)


def plot_rl_training(run_dirs: list[Path], out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.4))
    for i, d in enumerate(run_dirs):
        recs = [json.loads(line) for line in (d / "log.jsonl").read_text(encoding="utf-8").splitlines()]
        steps = [r["step"] for r in recs]
        color = plotting.COLORS[i % len(plotting.COLORS)]
        axes[0].plot(steps, [r["j"] for r in recs], color=color, label=d.name)
        axes[0].plot(steps, [r["j_star"] for r in recs], color=color, ls=":", lw=1)
        axes[1].plot(steps, [r["er"] for r in recs], color=color)
        axes[1].plot(steps, [r["er_star"] for r in recs], color=color, ls=":", lw=1)
        axes[2].plot(steps, [r["kl"] for r in recs], color=color)
        axes[2].plot(steps, [r["kl_star"] for r in recs], color=color, ls=":", lw=1)
    for ax, title in zip(axes, ["цель J = ΔE[r] − β·KL (пунктир — π*)", "ΔE[r] в единицах масштаба",
                                "KL(π ‖ π_ref)"]):
        ax.set_title(title)
        ax.set_xlabel("шаг")
    axes[0].legend(fontsize=7)
    plotting.save(fig, out)


def table(base, rows, rl_rows) -> str:
    cols = ["β", "KL", "NDCG@10", "Recall@100", "new NDCG@10", "ΔE proxy", "ΔE судьи", "длина, с", "из истории",
            "log10 попул.", "артистов", "coverage@100", "Jaccard"]
    lines = ["| политика | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]

    def line(name, r, reward):
        ls = r["lists"]
        dp = gain(r, base, "proxy", reward) if reward else 0.0
        dj = gain(r, base, "judge", reward) if reward else 0.0
        return (f"| {name} | {r.get('beta', '—')} | {r['kl']:.2f} | {r['all']['ndcg@10']:.3f} | "
                f"{r['all']['recall@100']:.3f} | {r['new']['ndcg@10']:.3f} | {dp:+.4g} | {dj:+.4g} | "
                f"{ls['track_length_sec@10']:.0f} | {ls['in_history@10']:.2f} | {ls['log10_train_count@10']:.2f} | "
                f"{ls['unique_artists@10']:.1f} | {ls['coverage@100']:.3f} | {ls['pair_jaccard@10']:.3f} |")

    lines.append(line("GPT-2 (π_ref)", base, None))
    for reward in REWARD_ORDER:
        for r in curve(rows, reward):
            if r["beta"] in (1, 0.3, 0.1, 0.02):
                lines.append(line(f"π* {reward}", r, reward))
        for r in rl_rows:
            if r["reward"] == reward:
                lines.append(line(r["label"], r, reward))
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sweep", default="reports/results/rl_sweep.json")
    p.add_argument("--rl", default="reports/results/rl_policies.json")
    args = p.parse_args()
    plotting.setup()
    base, rows = load(Path(args.sweep))
    rl_rows = []
    if Path(args.rl).exists():
        _, rl_all = load(Path(args.rl))
        meta = {r["name"]: r for r in rl_all if r.get("policy")}
        for name, r in meta.items():
            reward, label = name.split(":")[0], name.split(":")[1]
            rl_rows.append({**r, "reward": reward, "label": label})
    fig_dir = Path("reports/figures")
    plot_pareto(base, rows, rl_rows, fig_dir / "rl_pareto.png")
    plot_overoptimisation(base, rows, fig_dir / "rl_overoptimisation.png")
    plot_behaviour(base, rows, fig_dir / "rl_behaviour.png")
    plot_defences(base, rows, fig_dir / "rl_defences.png")
    run_dirs = sorted(d for d in Path("checkpoints/rl").glob("*")
                      if (d / "policy.pt").exists() and "reward" in json.loads((d / "args.json").read_text()))
    if run_dirs:
        plot_rl_training(run_dirs, fig_dir / "rl_training.png")
    md = table(base, rows, rl_rows)
    Path("reports/results/stage3_tables.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
