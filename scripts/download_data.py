"""Download Yambda-50M (multi_event) and the track -> artist mapping from Hugging Face."""
import argparse
import shutil
import urllib.request
from pathlib import Path

BASE_URL = "https://huggingface.co/datasets/yandex/yambda/resolve/main/"
FILES = [
    "flat/50m/multi_event.parquet",
    "artist_item_mapping.parquet",
]


def download(url: str, dst: Path) -> None:
    tmp = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as f:
        shutil.copyfileobj(resp, f, length=1 << 20)
    tmp.replace(dst)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/raw")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        dst = out / Path(name).name
        if dst.exists():
            print(f"skip {dst} (exists)")
            continue
        print(f"downloading {name} -> {dst}")
        download(BASE_URL + name, dst)
        print(f"  {dst.stat().st_size / 2**20:.1f} MB")


if __name__ == "__main__":
    main()
