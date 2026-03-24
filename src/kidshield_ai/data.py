from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd


RANDOM_SEED = 42


def normalize_text(value: str) -> str:
    return " ".join(str(value).strip().split())


def make_record(text: str, label: int, source: str, metadata: dict | None = None) -> dict:
    return {
        "text": normalize_text(text),
        "label": int(label),
        "source": source,
        "metadata": metadata or {},
    }


def load_jigsaw_toxic(raw_dir: Path) -> list[dict]:
    candidates = [
        raw_dir / "jigsaw-toxic" / "train.csv",
        raw_dir / "jigsaw-toxic" / "jigsaw-toxic-comment-train.csv",
    ]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        return []

    df = pd.read_csv(path)
    label_columns = [
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    ]
    missing = [column for column in label_columns + ["comment_text"] if column not in df.columns]
    if missing:
        raise ValueError(f"Jigsaw toxic file is missing columns: {missing}")

    records = []
    for row in df.itertuples(index=False):
        label = int(any(int(getattr(row, column)) > 0 for column in label_columns))
        text = getattr(row, "comment_text")
        records.append(
            make_record(
                text=text,
                label=label,
                source="jigsaw_toxic",
                metadata={"task": "toxicity_binary"},
            )
        )
    return records


def load_jigsaw_multilingual(raw_dir: Path) -> list[dict]:
    path = raw_dir / "jigsaw-multilingual" / "validation.csv"
    if not path.exists():
        return []

    df = pd.read_csv(path)
    missing = [column for column in ["comment_text", "toxic"] if column not in df.columns]
    if missing:
        raise ValueError(f"Jigsaw multilingual file is missing columns: {missing}")

    records = []
    for row in df.itertuples(index=False):
        metadata = {"task": "toxicity_binary"}
        if hasattr(row, "lang"):
            metadata["lang"] = getattr(row, "lang")
        records.append(
            make_record(
                text=getattr(row, "comment_text"),
                label=int(float(getattr(row, "toxic")) >= 0.5),
                source="jigsaw_multilingual",
                metadata=metadata,
            )
        )
    return records


def majority_label(annotators: list[dict]) -> str:
    labels = [entry["label"].strip().lower() for entry in annotators if "label" in entry]
    if not labels:
        raise ValueError("HateXplain sample has no annotator labels")
    counts = Counter(labels)
    return counts.most_common(1)[0][0]


def load_hatexplain(raw_dir: Path) -> tuple[list[dict], dict[str, list[str]]]:
    dataset_path = raw_dir / "hatexplain" / "dataset.json"
    split_path = raw_dir / "hatexplain" / "post_id_divisions.json"
    if not dataset_path.exists() or not split_path.exists():
        return [], {}

    with dataset_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    with split_path.open("r", encoding="utf-8") as handle:
        split_ids = json.load(handle)

    records = []
    for post_id, sample in payload.items():
        post_tokens = sample.get("post_tokens", [])
        text = " ".join(post_tokens)
        label_name = majority_label(sample.get("annotators", []))
        label = 0 if label_name == "normal" else 1
        records.append(
            make_record(
                text=text,
                label=label,
                source="hatexplain",
                metadata={
                    "task": "toxicity_binary",
                    "original_label": label_name,
                    "post_id": post_id,
                },
            )
        )
    return records, split_ids


def _read_nonempty_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip() and not line.startswith("#")]


def load_phishing_database(raw_dir: Path, limit: int | None = None) -> list[dict]:
    base_dir = raw_dir / "phishing-database"
    candidates = [
        base_dir / "links.list",
        base_dir / "domains.list",
        base_dir / "phishing-links-ACTIVE.txt",
        base_dir / "phishing-domains-ACTIVE.txt",
    ]
    records: list[dict] = []
    for path in candidates:
        if not path.exists():
            continue
        kind = "url" if "link" in path.name or path.name == "links.list" else "domain"
        for value in _read_nonempty_lines(path):
            records.append(
                make_record(
                    text=value,
                    label=1,
                    source="phishing_database",
                    metadata={"task": "phishing_binary", "kind": kind},
                )
            )
    if limit is not None and len(records) > limit:
        rng = random.Random(RANDOM_SEED)
        records = rng.sample(records, limit)
    return records


def load_majestic_million(raw_dir: Path, limit: int | None = None) -> list[dict]:
    path = raw_dir / "majestic-million" / "majestic_million.csv"
    if not path.exists():
        return []

    df = pd.read_csv(path)
    domain_column = None
    for candidate in ["Domain", "domain"]:
        if candidate in df.columns:
            domain_column = candidate
            break
    if domain_column is None:
        raise ValueError("Majestic Million file must contain a Domain column")

    if limit is not None:
        df = df.head(limit)

    records = []
    for row in df.itertuples(index=False):
        records.append(
            make_record(
                text=getattr(row, domain_column),
                label=0,
                source="majestic_million",
                metadata={"task": "phishing_binary", "kind": "domain"},
            )
        )
    return records


def train_val_test_split(records: list[dict], train_ratio: float = 0.8, val_ratio: float = 0.1) -> dict[str, list[dict]]:
    rng = random.Random(RANDOM_SEED)
    shuffled = list(records)
    rng.shuffle(shuffled)

    train_end = int(len(shuffled) * train_ratio)
    val_end = train_end + int(len(shuffled) * val_ratio)
    return {
        "train": shuffled[:train_end],
        "validation": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            serialized = {
                "text": record["text"],
                "label": record["label"],
                "source": record["source"],
            }
            handle.write(json.dumps(serialized, ensure_ascii=True) + "\n")


def summarize_records(records: list[dict]) -> dict:
    label_counts = Counter(record["label"] for record in records)
    source_counts = Counter(record["source"] for record in records)
    return {
        "rows": len(records),
        "label_distribution": {str(key): value for key, value in sorted(label_counts.items())},
        "source_distribution": dict(sorted(source_counts.items())),
    }
