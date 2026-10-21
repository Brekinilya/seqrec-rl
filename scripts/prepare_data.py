"""Build the vocabulary and the tokenized listen log shared by all later stages."""
import argparse
import json
from pathlib import Path

import polars as pl

from seqrec_rl.data import (N_SPECIAL, OOV, TemporalSplit, attach_feedback, build_vocab, listens,
                            load_events, tokenize)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--out", default="data/processed")
    parser.add_argument("--min-count", type=int, default=20, help="min train listens for a track to get a token")
    args = parser.parse_args()
    raw, out = Path(args.raw), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    events = load_events(raw / "multi_event.parquet")
    split = TemporalSplit.default()
    lis = attach_feedback(listens(events), events)

    artists = (pl.read_parquet(raw / "artist_item_mapping.parquet")
               .group_by("item_id").agg(pl.col("artist_id").min()))
    vocab = (build_vocab(split.train(lis), args.min_count)
             .join(artists, on="item_id", how="left", maintain_order="left"))
    lis = tokenize(lis, vocab).select(
        "uid", "timestamp", "item_id", "token", "is_organic", "played_ratio_pct",
        "track_length_seconds", "liked", "disliked")
    likes = (events.filter(pl.col("event_type") == "like")
             .select("uid", "timestamp", "item_id")
             .pipe(tokenize, vocab))

    vocab.write_parquet(out / "vocab.parquet")
    lis.write_parquet(out / "listens.parquet")
    likes.write_parquet(out / "likes.parquet")

    train = split.train(lis)
    meta = {
        "split": split.__dict__,
        "min_count": args.min_count,
        "n_items": vocab.height,
        "n_tokens": vocab.height + N_SPECIAL,
        "n_listens": lis.height,
        "train_listens": train.height,
        "train_oov_share": float((train["token"] == OOV).mean()),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
