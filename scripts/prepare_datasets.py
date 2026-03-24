#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kidshield_ai.data import (  # noqa: E402
    load_hatexplain,
    load_jigsaw_multilingual,
    load_jigsaw_toxic,
    load_majestic_million,
    load_phishing_database,
    summarize_records,
    train_val_test_split,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare merged data for DeBERTa fine-tuning.")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--majestic-limit", type=int, default=200000)
    parser.add_argument("--phishing-limit", type=int, default=200000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    combined_train: list[dict] = []
    combined_validation: list[dict] = []
    combined_test: list[dict] = []

    jigsaw_toxic = load_jigsaw_toxic(args.raw_dir)
    jigsaw_toxic_splits = train_val_test_split(jigsaw_toxic)

    jigsaw_multilingual = load_jigsaw_multilingual(args.raw_dir)
    jigsaw_multilingual_splits = train_val_test_split(jigsaw_multilingual)

    hatexplain_records, hatexplain_splits = load_hatexplain(args.raw_dir)
    hatexplain_by_id = {record["metadata"]["post_id"]: record for record in hatexplain_records}
    hatexplain_train = [hatexplain_by_id[post_id] for post_id in hatexplain_splits.get("train", []) if post_id in hatexplain_by_id]
    hatexplain_validation = [hatexplain_by_id[post_id] for post_id in hatexplain_splits.get("val", []) if post_id in hatexplain_by_id]
    hatexplain_test = [hatexplain_by_id[post_id] for post_id in hatexplain_splits.get("test", []) if post_id in hatexplain_by_id]

    phishing = load_phishing_database(args.raw_dir, limit=args.phishing_limit)
    phishing_splits = train_val_test_split(phishing)

    majestic = load_majestic_million(args.raw_dir, limit=args.majestic_limit)
    majestic_splits = train_val_test_split(majestic)

    for split_name, source_splits in [
        ("train", jigsaw_toxic_splits),
        ("train", jigsaw_multilingual_splits),
        ("train", phishing_splits),
        ("train", majestic_splits),
        ("validation", jigsaw_toxic_splits),
        ("validation", jigsaw_multilingual_splits),
        ("validation", phishing_splits),
        ("validation", majestic_splits),
        ("test", jigsaw_toxic_splits),
        ("test", jigsaw_multilingual_splits),
        ("test", phishing_splits),
        ("test", majestic_splits),
    ]:
        if split_name == "train":
            combined_train.extend(source_splits["train"])
        elif split_name == "validation":
            combined_validation.extend(source_splits["validation"])
        else:
            combined_test.extend(source_splits["test"])

    combined_train.extend(hatexplain_train)
    combined_validation.extend(hatexplain_validation)
    combined_test.extend(hatexplain_test)

    if not combined_train or not combined_validation:
        raise SystemExit("Not enough data found. Add raw dataset files under data/raw before preprocessing.")

    write_jsonl(args.output_dir / "train.jsonl", combined_train)
    write_jsonl(args.output_dir / "validation.jsonl", combined_validation)
    write_jsonl(args.output_dir / "test.jsonl", combined_test)

    report = {
        "train": summarize_records(combined_train),
        "validation": summarize_records(combined_validation),
        "test": summarize_records(combined_test),
        "raw_sources": {
            "jigsaw_toxic_rows": len(jigsaw_toxic),
            "jigsaw_multilingual_rows": len(jigsaw_multilingual),
            "hatexplain_rows": len(hatexplain_records),
            "phishing_rows": len(phishing),
            "majestic_rows": len(majestic),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "dataset_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
