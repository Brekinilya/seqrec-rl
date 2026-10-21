"""Tables and figures for stage 1: baselines vs the base GPT-2 model on test."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from seqrec_rl import plotting

REL_COLS = [("all", "listen_plus", "ndcg@10"), ("all", "listen_plus", "recall@10"), ("all", "listen_plus", "map@10"),
            ("all", "listen_plus", "ndcg@100"), ("all", "listen_plus", "recall@100"),
            ("new", "listen_plus", "ndcg@10"), ("new", "listen_plus", "recall@100"),
            ("all", "like", "ndcg@100"), ("all", "like", "recall@100")]
LIST_COLS = ["coverage@10", "coverage@100", "gini@10", "pair_jaccard@10", "in_history@10",
             "log10_train_count@10", "track_length_sec@10", "unique_artists@10"]


def table(rows: dict, cols, getter, fmt="{:.4f}") -> str:
    head = "| модель | " + " | ".join(cols) + " |"
    sep = "|---" * (len(cols) + 1) + "|"
    lines = [head, sep]
    for name, res in rows.items():
        lines.append(f"| {name} | " + " | ".join(fmt.format(getter(res, c)) for c in cols) + " |")
    return "\n".join(lines)


def plot_training(log_path: Path, out: Path) -> None:
    recs = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    train = [r for r in recs if "loss" in r]
    val = [r for r in recs if "val" in r]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.2))
    axes[0].plot([r["step"] for r in train], [r["loss"] for r in train], color=plotting.COLORS[0])
    axes[0].set_title("train loss (CE)")
    axes[0].set_xlabel("шаг")
    for ax, key, title in [(axes[1], "ndcg@10", "val NDCG@10, все треки"),
                           (axes[2], "new_ndcg@10", "val NDCG@10, новые треки")]:
        ax.plot([r["step"] for r in val], [r["val"][key] for r in val], marker="o", color=plotting.COLORS[0])
        ax.set_title(title)
        ax.set_xlabel("шаг")
    plotting.save(fig, out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="reports/results")
    p.add_argument("--log", default="checkpoints/base/log.jsonl")
    p.add_argument("--models", nargs="+", default=["GPT2"])
    args = p.parse_args()
    res_dir = Path(args.results)
    plotting.setup()

    rows = {}
    base = json.loads((res_dir / "baselines_test.json").read_text(encoding="utf-8"))
    rows.update({k: v for k, v in base.items()})
    for name in args.models:
        rows[name] = json.loads((res_dir / f"{name}_test.json").read_text(encoding="utf-8"))

    rel = table(rows, [f"{proto}:{tgt}:{m}" for proto, tgt, m in REL_COLS],
                lambda r, c: r[c.split(":")[0]][c.split(":")[1]][c.split(":")[2]])
    lists = {proto: table(rows, LIST_COLS, lambda r, c, proto=proto: r[proto]["lists"][c], "{:.3f}")
             for proto in ("all", "new")}
    md = (f"### Релевантность (test)\n\n{rel}\n\n### Свойства списков, протокол all\n\n{lists['all']}\n\n"
          f"### Свойства списков, протокол new\n\n{lists['new']}\n")
    (res_dir / "stage1_tables.md").write_text(md, encoding="utf-8")
    print(md)

    if Path(args.log).exists():
        plot_training(Path(args.log), Path("reports/figures/stage1_training.png"))


if __name__ == "__main__":
    main()
