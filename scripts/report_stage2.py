"""Markdown tables for the reward-model evaluation (reports/results/rm_eval.json)."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

from seqrec_rl import plotting

ORDER = ["global", "item", "user", "gbm", "scratch10", "m0", "m1", "m2", "ensemble"]
NAMES = {"global": "среднее", "item": "среднее по треку", "user": "среднее по пользователю", "gbm": "бустинг",
         "scratch10": "RM с нуля", "m0": "RM m0 (proxy)", "m1": "RM m1", "m2": "RM m2", "ensemble": "ансамбль m0–m2"}
COLS = [("ratio", "mae"), ("ratio", "spearman"), ("listen_plus", "roc_auc"), ("listen_plus", "bce"),
        ("listen_plus", "ece"), ("like", "roc_auc"), ("like", "pr_auc"), ("dislike", "roc_auc"),
        ("dislike", "pr_auc"), ("cont", "mae"), ("cont", "spearman")]


def cell(r: dict, head: str, metric: str) -> str:
    v = r.get(head, {}).get(metric)
    return "—" if v is None or v != v else f"{v:.3f}"


def table(res: dict, cols) -> str:
    head = "| модель | " + " | ".join(f"{h} {m}" for h, m in cols) + " |"
    lines = [head, "|---" * (len(cols) + 1) + "|"]
    for k in ORDER:
        if k in res:
            lines.append(f"| {NAMES[k]} | " + " | ".join(cell(res[k], h, m) for h, m in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    res = json.loads(Path("reports/results/rm_eval.json").read_text(encoding="utf-8"))
    parts = []
    for name in ("test", "val", "holdout"):
        r = res["results"][name]
        cols = COLS + ([("relisten", "roc_auc"), ("relisten", "pr_auc")] if name == "holdout" else [])
        parts.append(f"### {name}: {r['n_events']} событий, {r['n_users']} пользователей\n\n{table(r, cols)}\n")

    sl = res["slices"]["test"]
    cols = [("listen_plus", "roc_auc"), ("ratio", "mae"), ("like", "roc_auc"), ("like", "pr_auc"),
            ("like", "mean_label")]
    lines = ["| срез | модель | " + " | ".join(f"{h} {m}" for h, m in cols) + " |", "|---" * (len(cols) + 2) + "|"]
    for sname, by_model in sl.items():
        for k in ("gbm", "m0", "ensemble"):
            lines.append(f"| {sname} | {NAMES[k]} | " + " | ".join(cell(by_model[k], h, m) for h, m in cols) + " |")
    parts.append("### Срезы test\n\n" + "\n".join(lines) + "\n")

    dis = res["disagreement"]["test"]
    heads = list(next(iter(dis.values())).keys())
    lines = ["| срез | " + " | ".join(heads) + " |", "|---" * (len(heads) + 1) + "|"]
    for sname, v in dis.items():
        lines.append(f"| {sname} | " + " | ".join(f"{v[h]:.4f}" for h in heads) + " |")
    parts.append("### Разброс ансамбля (std предсказаний m0–m2), test\n\n" + "\n".join(lines) + "\n")

    prefs = json.loads(Path("reports/results/rm_preferences.json").read_text(encoding="utf-8"))
    base = json.loads(Path("reports/results/GPT2_test.json").read_text(encoding="utf-8"))
    head = ("| список | NDCG@10 | new NDCG@10 | длина, с | из истории | log10 попул. | артистов | "
            "награда (proxy) | награда (судьи) | награда GPT-2 (proxy) | разброс ансамбля | разброс на GPT-2 |")
    lines = [head, "|---" * 12 + "|"]
    bl = base["all"]["lists"]
    lines.append(f"| GPT-2 | {base['all']['listen_plus']['ndcg@10']:.3f} | {base['new']['listen_plus']['ndcg@10']:.3f} | "
                 f"{bl['track_length_sec@10']:.0f} | {bl['in_history@10']:.2f} | {bl['log10_train_count@10']:.2f} | "
                 f"{bl['unique_artists@10']:.1f} | — | — | — | — | — |")
    for r, v in prefs.items():
        ls = v["all"]["lists"]
        lines.append(
            f"| greedy {r} | {v['all']['listen_plus']['ndcg@10']:.3f} | {v['new']['listen_plus']['ndcg@10']:.3f} | "
            f"{ls['track_length_sec@10']:.0f} | {ls['in_history@10']:.2f} | {ls['log10_train_count@10']:.2f} | "
            f"{ls['unique_artists@10']:.1f} | {v['reward@10']['greedy']:.4g} | {v['judge_reward@10']['greedy']:.4g} | "
            f"{v['reward@10']['GPT2']:.4g} | {v['ensemble_std@10']['greedy']:.3g} | {v['ensemble_std@10']['GPT2']:.3g} |")
    parts.append("### Жадные списки по reward model (test)\n\n" + "\n".join(lines) + "\n")

    md = "\n".join(parts)
    Path("reports/results/stage2_tables.md").write_text(md, encoding="utf-8")
    print(md)
    plot_greedy(prefs, bl)


def plot_greedy(prefs: dict, base_lists: dict) -> None:
    plotting.setup()
    show = ["ratio", "seconds", "like", "cont", "relisten"]
    panels = [("track_length_sec@10", "длина трека в top-10, с (лог. шкала)"),
              ("in_history@10", "доля треков из истории"), ("log10_train_count@10", "log10 популярности")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    for ax, (key, title) in zip(axes, panels):
        vals = [base_lists[key]] + [prefs[r]["all"]["lists"][key] for r in show]
        ax.bar(range(len(vals)), vals, color=[plotting.MUTED] + plotting.COLORS[:len(show)], width=0.7)
        ax.set_xticks(range(len(vals)), ["GPT-2"] + [f"greedy {r}" for r in show], rotation=30)
        ax.set_title(title)
        if key.startswith("track_length"):
            ax.set_yscale("log")
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.2f}" if v < 10 else f"{v:.0f}", ha="center", va="bottom", fontsize=7,
                    color=plotting.INK_2)
    plotting.save(fig, Path("reports/figures/rm_greedy_lists.png"))


if __name__ == "__main__":
    main()
