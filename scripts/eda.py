"""Exploratory analysis of Yambda-50M.

Writes figures to reports/figures and all numbers to reports/eda_stats.json.
"""
import argparse
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.ticker import PercentFormatter

from seqrec_rl import plotting
from seqrec_rl.data import DAY, LISTEN_PLUS_PCT, SESSION_GAP, TemporalSplit, listens, load_events

LENGTH_EDGES = [0, 120, 150, 180, 210, 240, 300, 420, 600, 10**7]
LENGTH_LABELS = ["<2", "2-2.5", "2.5-3", "3-3.5", "3.5-4", "4-5", "5-7", "7-10", ">10"]
RATIO_EDGES = [0, 10, 30, 50, 80, 100, 101, 10**4]
RATIO_LABELS = ["<10", "10-30", "30-50", "50-80", "80-99", "100", ">100"]
VOCAB_THRESHOLDS = [1, 2, 5, 10, 20, 50, 100]


def quantiles(s: pl.Series, qs=(0.1, 0.25, 0.5, 0.75, 0.9, 0.99)) -> dict:
    return {f"q{int(q * 100)}": float(s.quantile(q)) for q in qs} | {"mean": float(s.mean())}


def bucket(col: str, edges: list) -> pl.Expr:
    """Index of the bin [edges[i], edges[i + 1]) the value falls into."""
    expr = pl.lit(len(edges) - 2)
    for i in range(len(edges) - 3, -1, -1):
        expr = pl.when(pl.col(col) < edges[i + 1]).then(pl.lit(i)).otherwise(expr)
    return expr


def with_labels(t: pl.DataFrame, col: str, labels: list) -> pl.DataFrame:
    return t.with_columns(pl.Series(col, [labels[i] for i in t[col]]))


def match_feedback(lis: pl.DataFrame, events: pl.DataFrame, kind: str) -> pl.DataFrame:
    """Match each like/dislike to the latest listen of the same track at or before it."""
    fb = (events.filter(pl.col("event_type") == kind)
          .select("uid", "item_id", pl.col("timestamp").alias("fb_ts"))
          .sort("fb_ts"))
    right = lis.select("uid", "item_id", "timestamp", "track_length_seconds").sort("timestamp")
    with warnings.catch_warnings():
        # both sides are sorted by time globally, which is what join_asof needs
        warnings.filterwarnings("ignore", message="Sortedness of columns")
        m = fb.join_asof(right, left_on="fb_ts", right_on="timestamp", by=["uid", "item_id"],
                         strategy="backward")
    return m.with_columns(delay_sec=pl.col("fb_ts") - pl.col("timestamp"))


def feedback_delay_stats(m: pl.DataFrame) -> dict:
    n = m.height
    matched = m.filter(pl.col("timestamp").is_not_null())
    d = matched["delay_sec"]
    within_track = (matched["delay_sec"] <= matched["track_length_seconds"] + 60).sum()
    return {
        "n": n,
        "share_without_prior_listen": 1 - matched.height / n,
        "share_within_track_plus_1min": within_track / n,
        "share_within_1h": (d <= 3600).sum() / n,
        "delay_sec": quantiles(d.cast(pl.Float64)),
    }


def flag_listens(lis: pl.DataFrame, m: pl.DataFrame, name: str) -> pl.DataFrame:
    hits = (m.filter(pl.col("timestamp").is_not_null()
                     & (pl.col("delay_sec") <= pl.col("track_length_seconds") + 60))
            .select("uid", "item_id", "timestamp").unique()
            .with_columns(pl.lit(True).alias(name)))
    return (lis.join(hits, on=["uid", "item_id", "timestamp"], how="left")
            .with_columns(pl.col(name).fill_null(False)))


def plot_daily(lis: pl.DataFrame, split: TemporalSplit, out: Path) -> None:
    daily = lis.group_by((pl.col("timestamp") // DAY).alias("day")).len().sort("day")
    t0 = split.train_end - 2 * DAY
    hourly = (lis.filter(pl.col("timestamp") >= t0)
              .group_by(((pl.col("timestamp") - t0) // 3600).alias("h")).len().sort("h"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.2), gridspec_kw={"width_ratios": [2, 1]})
    ax = axes[0]
    ax.plot(daily["day"], daily["len"] / 1e3, color=plotting.COLORS[0])
    ax.set_xlabel("день")
    ax.set_ylabel("прослушиваний, тыс.")
    ax.set_title("Активность по дням")

    ax = axes[1]
    ax.plot(hourly["h"] / 24 + t0 / DAY, hourly["len"] / 1e3, color=plotting.COLORS[0])
    for start, end, label, color in [(split.val_start, split.val_end, "val", plotting.COLORS[2]),
                                     (split.test_start, lis["timestamp"].max(), "test", plotting.COLORS[1])]:
        ax.axvspan(start / DAY, end / DAY, color=color, alpha=0.12, lw=0)
        ax.text((start + end) / 2 / DAY, 0.97, label, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=8, color=plotting.INK_2)
    ax.set_xlabel("день")
    ax.set_ylabel("прослушиваний в час, тыс.")
    ax.set_title("Последние дни и разбиение")
    plotting.save(fig, out / "daily_activity.png")


def plot_ratio_hist(lis: pl.DataFrame, out: Path) -> None:
    r = lis["played_ratio_pct"].clip(0, 110).to_numpy()
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(r, bins=np.arange(0, 112, 2), color=plotting.COLORS[0], rwidth=0.9)
    ax.axvline(LISTEN_PLUS_PCT, color=plotting.MUTED, lw=1, ls="--")
    ax.set_yscale("log")
    ax.set_xlabel("played_ratio_pct (значения >110 обрезаны)")
    ax.set_ylabel("прослушиваний")
    ax.set_title("Распределение доли дослушивания")
    plotting.save(fig, out / "played_ratio_hist.png")


def by_length(lis: pl.DataFrame) -> pl.DataFrame:
    t = (lis.with_columns(len_bin=bucket("track_length_seconds", LENGTH_EDGES))
         .group_by("len_bin", "is_repeat")
         .agg(n=pl.len(),
              ratio=pl.col("played_ratio_pct").clip(0, 100).mean(),
              listen_plus=(pl.col("played_ratio_pct") >= LISTEN_PLUS_PCT).mean(),
              seconds=(pl.col("played_ratio_pct").clip(0, 100) / 100 * pl.col("track_length_seconds")).mean(),
              like=pl.col("liked").mean())
         .sort("len_bin", "is_repeat"))
    return with_labels(t, "len_bin", LENGTH_LABELS)


def plot_by_length(t: pl.DataFrame, out: Path) -> None:
    panels = [("ratio", "средняя доля, %"), ("listen_plus", "доля Listen+"),
              ("seconds", "прослушано, сек"), ("like", "доля лайков")]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
    x = np.arange(len(LENGTH_LABELS))
    for ax, (col, title) in zip(axes, panels):
        for i, (rep, name) in enumerate([(False, "первое прослушивание"), (True, "повтор")]):
            part = t.filter(pl.col("is_repeat") == rep)
            ax.plot(x, part[col], marker="o", color=plotting.COLORS[i], label=name)
        ax.set_xticks(x, LENGTH_LABELS, rotation=45)
        ax.set_xlabel("длина трека, мин")
        ax.set_title(title)
    axes[0].legend()
    plotting.save(fig, out / "response_vs_length.png")


def by_group(lis: pl.DataFrame, key: str) -> pl.DataFrame:
    return (lis.group_by(key)
            .agg(n=pl.len(),
                 ratio=pl.col("played_ratio_pct").clip(0, 100).mean(),
                 listen_plus=(pl.col("played_ratio_pct") >= LISTEN_PLUS_PCT).mean(),
                 like=pl.col("liked").mean(),
                 dislike=pl.col("disliked").mean(),
                 session_end=pl.col("session_end").mean())
            .with_columns(share=pl.col("n") / pl.col("n").sum())
            .sort(key))


def plot_groups(org: pl.DataFrame, rep: pl.DataFrame, out: Path) -> None:
    panels = [("listen_plus", "доля Listen+"), ("like", "доля лайков"),
              ("dislike", "доля дизлайков"), ("session_end", "P(конец сессии)")]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    labels = ["рекоменд.", "органика", "первое", "повтор"]
    x = [0, 1, 2.5, 3.5]
    for ax, (col, title) in zip(axes, panels):
        vals = list(org[col]) + list(rep[col])
        ax.bar(x, vals, color=plotting.COLORS[:4], width=0.8)
        ax.set_xticks(x, labels, rotation=30)
        ax.set_title(title)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        for xi, v in zip(x, vals):
            ax.text(xi, v, f"{v:.2%}", ha="center", va="bottom", fontsize=7, color=plotting.INK_2)
    plotting.save(fig, out / "organic_and_repeat.png")


def plot_sessions(lis: pl.DataFrame, out: Path) -> dict:
    gaps = lis["dt_prev"].drop_nulls()
    gaps = gaps.filter(gaps > 0).cast(pl.Float64)
    returns = lis.filter(pl.col("session_end"))["dt_next"].cast(pl.Float64) / 3600

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
    ax = axes[0]
    ax.hist(np.log10(gaps.to_numpy()), bins=120, color=plotting.COLORS[0])
    ax.axvline(np.log10(SESSION_GAP), color=plotting.MUTED, lw=1, ls="--")
    ticks = [10, 60, 600, 3600, 86400, 7 * 86400]
    ax.set_xticks(np.log10(ticks), ["10с", "1м", "10м", "1ч", "1д", "7д"])
    ax.set_yscale("log")
    ax.set_title("Пауза между соседними прослушиваниями")

    ax = axes[1]
    r = np.sort(returns.to_numpy())
    ax.plot(r, np.arange(1, len(r) + 1) / len(r), color=plotting.COLORS[0])
    ax.set_xscale("log")
    ax.set_xlim(0.5, 30 * 24)
    ax.set_xticks([0.5, 1, 6, 24, 72, 168, 720], ["30м", "1ч", "6ч", "1д", "3д", "7д", "30д"])
    ax.set_title("Время до следующей сессии (CDF)")
    plotting.save(fig, out / "sessions.png")

    sess_len = lis.group_by("uid", "session_idx").len()["len"]
    return {
        "n_sessions": sess_len.len(),
        "session_length": quantiles(sess_len.cast(pl.Float64)),
        "return_within": {h: float((returns <= hours).mean())
                          for h, hours in [("1h", 1), ("6h", 6), ("24h", 24), ("72h", 72), ("7d", 168)]},
        "return_hours": quantiles(returns),
    }


def by_ratio(lis: pl.DataFrame) -> pl.DataFrame:
    t = (lis.filter(pl.col("dt_next").is_not_null())
         .with_columns(r_bin=bucket("played_ratio_pct", RATIO_EDGES))
         .group_by("r_bin")
         .agg(n=pl.len(), session_end=pl.col("session_end").mean(), like=pl.col("liked").mean())
         .sort("r_bin"))
    return with_labels(t, "r_bin", RATIO_LABELS)


def plot_by_ratio(t: pl.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    x = np.arange(t.height)
    for ax, (col, title) in zip(axes, [("session_end", "P(сессия закончилась после трека)"),
                                       ("like", "доля лайков")]):
        ax.bar(x, t[col], color=plotting.COLORS[0], width=0.7)
        ax.set_xticks(x, t["r_bin"].to_list())
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
        ax.set_xlabel("played_ratio_pct")
        ax.set_title(title)
    plotting.save(fig, out / "response_vs_ratio.png")


def vocab_coverage(train: pl.DataFrame, targets: pl.DataFrame) -> tuple[list, pl.DataFrame]:
    counts = train.group_by("item_id").len().sort("len", descending=True)
    total = counts["len"].sum()
    tgt = targets.join(counts, on="item_id", how="left").with_columns(pl.col("len").fill_null(0))
    rows = []
    for k in VOCAB_THRESHOLDS:
        keep = counts.filter(pl.col("len") >= k)
        rows.append({
            "min_train_listens": k,
            "n_items": keep.height,
            "train_listen_coverage": keep["len"].sum() / total,
            "test_target_coverage": float((tgt["len"] >= k).mean()),
            "test_new_target_coverage": float(tgt.filter(~pl.col("seen"))["len"].ge(k).mean()),
        })
    return rows, counts


def plot_popularity(counts: pl.DataFrame, cov: list, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    c = counts["len"].to_numpy()
    axes[0].loglog(np.arange(1, len(c) + 1), c, color=plotting.COLORS[0])
    axes[0].set_xlabel("ранг трека")
    axes[0].set_ylabel("прослушиваний в train")
    axes[0].set_title("Популярность треков")
    ks = [r["min_train_listens"] for r in cov]
    axes[1].plot(ks, [r["train_listen_coverage"] for r in cov], marker="o", label="прослушивания train")
    axes[1].plot(ks, [r["test_target_coverage"] for r in cov], marker="o", label="таргеты test")
    axes[1].plot(ks, [r["test_new_target_coverage"] for r in cov], marker="o", label="новые таргеты test")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("мин. число прослушиваний трека в train")
    axes[1].set_title("Покрытие словарём")
    axes[1].legend()
    plotting.save(fig, out / "popularity_and_vocab.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw")
    parser.add_argument("--out", default="reports")
    args = parser.parse_args()
    out, fig_dir = Path(args.out), Path(args.out) / "figures"
    plotting.setup()

    events = load_events(Path(args.data) / "multi_event.parquet")
    events = events.with_columns(pl.col("timestamp").cast(pl.Int32))
    split = TemporalSplit.default()
    stats = {
        "schema": {k: str(v) for k, v in events.schema.items()},
        "split": split.__dict__ | {"days": {k: v / DAY for k, v in split.__dict__.items()}},
        "span_days": (int(events["timestamp"].max()) - int(events["timestamp"].min())) / DAY,
        "events_by_type": dict(events.group_by("event_type").len().iter_rows()),
        "n_users": events["uid"].n_unique(),
        "n_items": events["item_id"].n_unique(),
    }

    lis = (listens(events)
           .with_columns(pl.col("played_ratio_pct").cast(pl.Int32),
                         pl.col("track_length_seconds").cast(pl.Int32))
           .sort("uid", "timestamp"))
    stats["n_listens"] = lis.height
    stats["n_listened_items"] = lis["item_id"].n_unique()
    stats["played_ratio"] = {
        "null_share": lis["played_ratio_pct"].null_count() / lis.height,
        "listen_plus_share": float((lis["played_ratio_pct"] >= LISTEN_PLUS_PCT).mean()),
        "over_100_share": float((lis["played_ratio_pct"] > 100).mean()),
        "eq_100_share": float((lis["played_ratio_pct"] == 100).mean()),
        "under_10_share": float((lis["played_ratio_pct"] < 10).mean()),
        "dist": quantiles(lis["played_ratio_pct"].cast(pl.Float64)),
    }
    stats["track_length_seconds"] = quantiles(lis["track_length_seconds"].cast(pl.Float64))
    per_item_len = lis.group_by("item_id").agg(pl.col("track_length_seconds").n_unique().alias("k"))
    stats["items_with_varying_length_share"] = float((per_item_len["k"] > 1).mean())

    like_m = match_feedback(lis, events, "like")
    dislike_m = match_feedback(lis, events, "dislike")
    stats["like_matching"] = feedback_delay_stats(like_m)
    stats["dislike_matching"] = feedback_delay_stats(dislike_m)
    lis = flag_listens(lis, like_m, "liked")
    lis = flag_listens(lis, dislike_m, "disliked")
    del like_m, dislike_m

    lis = lis.sort("uid", "timestamp").with_columns(
        dt_prev=pl.col("timestamp").diff().over("uid"),
        dt_next=pl.col("timestamp").shift(-1).over("uid") - pl.col("timestamp"),
        pos=pl.int_range(pl.len()).over("uid"),
    )
    lis = lis.with_columns(
        session_idx=(pl.col("dt_prev").is_null() | (pl.col("dt_prev") > SESSION_GAP)).cum_sum().over("uid"),
        session_end=pl.col("dt_next") > SESSION_GAP,
        # distance (in listens) to the previous listen of the same track
        repeat_dist=pl.col("pos") - pl.col("pos").shift(1).over("uid", "item_id"),
    ).with_columns(is_repeat=pl.col("repeat_dist").is_not_null())

    stats["listens_per_user"] = quantiles(lis.group_by("uid").len()["len"].cast(pl.Float64))
    stats["repeat_share"] = float(lis["is_repeat"].mean())
    dist = lis["repeat_dist"].drop_nulls()
    stats["repeat_distance"] = {
        "dist": quantiles(dist.cast(pl.Float64)),
        "consecutive_share_of_listens": float((dist == 1).sum() / lis.height),
        **{f"within_{k}_share_of_listens": float((dist <= k).sum() / lis.height) for k in (10, 50, 256, 1024)},
    }
    stats["organic_x_repeat"] = (lis.group_by("is_organic", "is_repeat")
                                 .agg(n=pl.len(), listen_plus=(pl.col("played_ratio_pct") >= LISTEN_PLUS_PCT).mean())
                                 .with_columns(share=pl.col("n") / pl.col("n").sum())
                                 .sort("is_organic", "is_repeat").to_dicts())
    stats["by_organic"] = by_group(lis, "is_organic").to_dicts()
    stats["by_repeat"] = by_group(lis, "is_repeat").to_dicts()
    stats["like_rate"] = float(lis["liked"].mean())
    stats["dislike_rate"] = float(lis["disliked"].mean())
    stats["sessions"] = plot_sessions(lis, fig_dir)
    length_table = by_length(lis)
    stats["by_length"] = length_table.to_dicts()
    ratio_table = by_ratio(lis)
    stats["by_ratio"] = ratio_table.to_dicts()
    stats["corr_ratio_length"] = float(
        lis.select(pl.corr(pl.col("played_ratio_pct").clip(0, 100), "track_length_seconds", method="spearman")).item())

    # Test-window targets: Listen+ events; "seen" = the user listened to the track before the test.
    first_seen = lis.group_by("uid", "item_id").agg(pl.col("timestamp").min().alias("first_ts"))
    test_lis = split.test(lis)
    targets = (test_lis.filter(pl.col("played_ratio_pct") >= LISTEN_PLUS_PCT)
               .join(first_seen, on=["uid", "item_id"])
               .with_columns(seen=pl.col("first_ts") < split.val_end))
    hist_users = split.history_for_test(lis)["uid"].unique()
    stats["test"] = {
        "n_listens": test_lis.height,
        "n_users_with_listens": test_lis["uid"].n_unique(),
        "n_listen_plus_targets": targets.height,
        "n_users_with_targets": targets["uid"].n_unique(),
        "cold_target_users": targets.filter(~pl.col("uid").is_in(hist_users.implode()))["uid"].n_unique(),
        "targets_per_user": quantiles(targets.group_by("uid").len()["len"].cast(pl.Float64)),
        "target_seen_share": float(targets["seen"].mean()),
        "target_organic_share": float(targets["is_organic"].cast(pl.Float64).mean()),
        "n_like_events": split.test(events).filter(pl.col("event_type") == "like").height,
    }
    stats["val"] = {
        "n_listens": split.val(lis).height,
        "n_users_with_listens": split.val(lis)["uid"].n_unique(),
    }

    train_lis = split.train(lis)
    cov, counts = vocab_coverage(train_lis, targets)
    stats["vocab_coverage"] = cov
    stats["train_listens_per_user"] = quantiles(train_lis.group_by("uid").len()["len"].cast(pl.Float64))

    artist_path = Path(args.data) / "artist_item_mapping.parquet"
    if artist_path.exists():
        artists = pl.read_parquet(artist_path)
        stats["artist_schema"] = {k: str(v) for k, v in artists.schema.items()}
        per_item = artists.group_by("item_id").len()["len"]
        first_artist = artists.group_by("item_id").agg(pl.col("artist_id").first())
        a = lis.select("item_id").join(first_artist, on="item_id", how="left")
        stats["artists"] = {
            "n_artists_mapped": first_artist["artist_id"].n_unique(),
            "items_with_multiple_artists_share": float((per_item > 1).mean()),
            "listen_coverage": float(a["artist_id"].is_not_null().mean()),
            "n_artists_in_listens": a["artist_id"].n_unique(),
        }

    plot_daily(lis, split, fig_dir)
    plot_ratio_hist(lis, fig_dir)
    plot_by_length(length_table, fig_dir)
    plot_groups(by_group(lis, "is_organic"), by_group(lis, "is_repeat"), fig_dir)
    plot_by_ratio(ratio_table, fig_dir)
    plot_popularity(counts, cov, fig_dir)

    out.mkdir(parents=True, exist_ok=True)
    (out / "eda_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False, default=str),
                                        encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
