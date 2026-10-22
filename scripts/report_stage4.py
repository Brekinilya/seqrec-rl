"""Figures and tables for stage 4 from reports/results/rollouts.json."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

from seqrec_rl import plotting

GREEDY = ["GPT-2", "π* ratio β=1", "π* ratio β=0.3", "π* ratio β=0.3, оптимистичный отклик", "RL ratio β=0.3",
          "Positive-CE"]
MODES = {"unknown": "отклик «неизвестно»", "expected": "ожидаемый отклик", "optimistic": "оптимистичный отклик"}


def plot_drift(res: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    for i, name in enumerate(GREEDY):
        steps = res[f"{name}|greedy"]["steps"]
        ks = [s["k"] for s in steps]
        ratio = [s["gen"]["ndcg@10"] / s["real"]["ndcg@10"] for s in steps]
        c = plotting.COLORS[i]
        axes[0].plot(ks, ratio, marker="o", color=c, label=name)
        axes[1].plot(ks, [s["kl_gen_real"] for s in steps], marker="o", color=c)
        axes[2].plot(ks, [s["jaccard_gen_real"] for s in steps], marker="o", color=c)
    titles = ["NDCG@10: сгенерированный контекст / реальный", "KL(π | сгенер. ‖ π | реальн.)",
              "пересечение top-10 (Jaccard)"]
    for ax, t in zip(axes, titles):
        ax.set_title(t)
        ax.set_xlabel("шаг генерации k")
    axes[0].legend(fontsize=7)
    plotting.save(fig, out)


def plot_reward(res: dict, out: Path) -> None:
    names = ["GPT-2", "π* ratio β=0.3", "π* ratio β=0.3, оптимистичный отклик"]
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.6))
    for ax, name in zip(axes, names):
        steps = res[f"{name}|greedy"]["steps"][:-1]
        ks = [s["k"] + 1 for s in steps]
        for i, (m, label) in enumerate(MODES.items()):
            ax.plot(ks, [s["reward"][m]["proxy"] for s in steps], marker="o", color=plotting.COLORS[i],
                    label=f"proxy, {label}")
            ax.plot(ks, [s["reward"][m]["judge"] for s in steps], marker="o", ls="--", color=plotting.COLORS[i],
                    label=f"судьи, {label}")
        ax.set_title(name)
        ax.set_xlabel("шаг k")
        ax.set_ylabel("E[доля дослушивания] трека a_k")
    axes[0].legend(fontsize=6)
    ax = axes[3]
    for i, name in enumerate(GREEDY):
        steps = res[f"{name}|greedy"]["steps"][:-1]
        ax.plot([s["k"] + 1 for s in steps], [s["reward"]["unknown"]["std"] for s in steps], marker="o",
                color=plotting.COLORS[i], label=name)
    ax.set_title("разброс ансамбля m0–m2 на треке a_k")
    ax.set_xlabel("шаг k")
    ax.legend(fontsize=6)
    plotting.save(fig, out)


def plot_items(res: dict, out: Path) -> None:
    panels = [("item_hit_test_listen_plus", "трек a_k дослушан в тестовый день"),
              ("item_in_history", "трек a_k есть в истории"), ("same_artist_as_prev", "тот же артист, что a_{k−1}"),
              ("loop_attempt", "argmax — уже сгенерированный трек"), ("item_length_sec", "длина a_k, с")]
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.4))
    for ax, (key, title) in zip(axes, panels):
        for i, name in enumerate(GREEDY):
            steps = [s for s in res[f"{name}|greedy"]["steps"][:-1] if s.get(key) is not None]
            ax.plot([s["k"] + 1 for s in steps], [s[key] for s in steps], marker="o", color=plotting.COLORS[i],
                    label=name)
        ax.set_title(title)
        ax.set_xlabel("шаг k")
    axes[0].legend(fontsize=6)
    plotting.save(fig, out)


def tables(res: dict) -> str:
    lines = ["| политика | декодирование | NDCG@10 k=0 | k=5 (сгенер.) | k=5 (реальн.) | KL k=5 | попадание a_k (k=1→5) | "
             "тот же артист k=2→5 | повтор argmax k=2→5 | награда a_k proxy k=1→5 | судьи k=1→5 |",
             "|---" * 11 + "|"]
    for key, r in res.items():
        if key == "real_items":
            continue
        name, dec = key.split("|")
        s = r["steps"]
        lines.append(
            f"| {name} | {dec} | {s[0]['gen']['ndcg@10']:.3f} | {s[-1]['gen']['ndcg@10']:.3f} | "
            f"{s[-1]['real']['ndcg@10']:.3f} | {s[-1]['kl_gen_real']:.2f} | "
            f"{s[0]['item_hit_test_listen_plus']:.3f} → {s[-2]['item_hit_test_listen_plus']:.3f} | "
            f"{s[1]['same_artist_as_prev']:.2f} → {s[-2]['same_artist_as_prev']:.2f} | "
            f"{s[1]['loop_attempt']:.2f} → {s[-2]['loop_attempt']:.2f} | "
            f"{s[0]['reward']['unknown']['proxy']:.3f} → {s[-2]['reward']['unknown']['proxy']:.3f} | "
            f"{s[0]['reward']['unknown']['judge']:.3f} → {s[-2]['reward']['unknown']['judge']:.3f} |")
    slate = ["| политика | декодирование | Top-K NDCG@5 | Next-K NDCG@5 | Top-K HitRate@5 | Next-K HitRate@5 | "
             "артистов Top-K | артистов Next-K | награда Top-K (судьи) | награда Next-K (судьи) |", "|---" * 10 + "|"]
    for key, r in res.items():
        if key == "real_items":
            continue
        name, dec = key.split("|")
        sl = r["slate"]
        slate.append(f"| {name} | {dec} | {sl['top_k']['ndcg@5']:.3f} | {sl['next_k']['ndcg@5']:.3f} | "
                     f"{sl['top_k']['hitrate@5']:.3f} | {sl['next_k']['hitrate@5']:.3f} | {sl['top_k_artists']:.2f} | "
                     f"{sl['next_k_artists']:.2f} | {sl['top_k_reward']['judge']:.3f} | "
                     f"{sl['next_k_reward']['unknown']['judge']:.3f} |")
    ri = res["real_items"]
    real = (f"Реальные первые 5 треков тестового дня ({ri['n']}): фактическая доля дослушивания {ri['actual_mean']:.3f}, "
            f"proxy предсказывает {ri['proxy_mean']:.3f}, судьи {ri['judge_mean']:.3f}, MAE proxy {ri['proxy_mae']:.3f}.")
    return "### По шагам\n\n" + "\n".join(lines) + "\n\n### Слейт из 5 треков\n\n" + "\n".join(slate) + "\n\n" + real + "\n"


def main() -> None:
    plotting.setup()
    res = json.loads(Path("reports/results/rollouts.json").read_text(encoding="utf-8"))
    fig_dir = Path("reports/figures")
    plot_drift(res, fig_dir / "multistep_drift.png")
    plot_reward(res, fig_dir / "multistep_reward.png")
    plot_items(res, fig_dir / "multistep_items.png")
    md = tables(res)
    Path("reports/results/stage4_tables.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
