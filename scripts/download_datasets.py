#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, destination.open("wb") as handle:
        handle.write(response.read())


def download_jigsaw_toxic(raw_dir: Path) -> None:
    output = raw_dir / "jigsaw-toxic" / "train.csv"
    url = "https://huggingface.co/datasets/thesofakillers/jigsaw-toxic-comment-classification-challenge/resolve/main/train.csv?download=true"
    download_file(url, output)
    print(f"saved {output}")


def download_hatexplain(raw_dir: Path) -> None:
    base = raw_dir / "hatexplain"
    files = {
        "dataset.json": "https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data/dataset.json",
        "post_id_divisions.json": "https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data/post_id_divisions.json",
    }
    for name, url in files.items():
        destination = base / name
        download_file(url, destination)
        print(f"saved {destination}")


def download_phishing_database(raw_dir: Path) -> None:
    base = raw_dir / "phishing-database"
    files = {
        "phishing-links-ACTIVE.txt": "https://raw.githubusercontent.com/Phishing-Database/Phishing.Database/master/phishing-links-ACTIVE.txt",
        "phishing-domains-ACTIVE.txt": "https://raw.githubusercontent.com/Phishing-Database/Phishing.Database/master/phishing-domains-ACTIVE.txt",
    }
    for name, url in files.items():
        destination = base / name
        download_file(url, destination)
        print(f"saved {destination}")


def download_majestic(raw_dir: Path) -> None:
    url = "https://raw.githubusercontent.com/kyle-w-brown/majestic_million/main/majestic_million.csv"
    destination = raw_dir / "majestic-million" / "majestic_million.csv"
    download_file(url, destination)
    print(f"saved {destination}")


def download_jigsaw_multilingual_placeholder(raw_dir: Path) -> None:
    destination = raw_dir / "jigsaw-multilingual" / "README.txt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "Exact Kaggle competition files for Jigsaw Multilingual Toxic Comment Classification were not publicly downloadable in this environment.\n"
        "Expected file for preprocessing: validation.csv\n",
        encoding="utf-8",
    )
    print(f"wrote placeholder {destination}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download public Kidshield dataset sources and mirrors.")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    unresolved: list[str] = []

    tasks = [
        ("jigsaw-toxic", download_jigsaw_toxic),
        ("hatexplain", download_hatexplain),
        ("phishing-database", download_phishing_database),
        ("majestic-million", download_majestic),
    ]
    for name, task in tasks:
        try:
            task(args.raw_dir)
        except Exception as exc:  # noqa: BLE001
            unresolved.append(f"{name}: {exc}")
            print(f"failed {name}: {exc}", file=sys.stderr)

    download_jigsaw_multilingual_placeholder(args.raw_dir)
    if unresolved:
        (args.raw_dir / "DOWNLOAD_ERRORS.txt").write_text("\n".join(unresolved) + "\n", encoding="utf-8")
        print(f"wrote {args.raw_dir / 'DOWNLOAD_ERRORS.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
