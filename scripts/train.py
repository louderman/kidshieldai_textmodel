#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import evaluate
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

HF_HOME = ROOT / ".cache" / "huggingface"
HF_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("HF_DATASETS_CACHE", str(HF_HOME / "datasets"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_HOME / "transformers"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune mDeBERTa on the processed Kidshield dataset.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="microsoft/mdeberta-v3-base")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--optim", default="adamw_torch")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-validation-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_files = {
        "train": str(args.data_dir / "train.jsonl"),
        "validation": str(args.data_dir / "validation.jsonl"),
        "test": str(args.data_dir / "test.jsonl"),
    }
    for split_name, path in data_files.items():
        if not Path(path).exists():
            raise SystemExit(f"Missing {split_name} split at {path}. Run scripts/prepare_datasets.py first.")

    dataset = load_dataset("json", data_files=data_files, cache_dir=str(HF_HOME / "datasets"))
    if args.max_train_samples is not None:
        dataset["train"] = dataset["train"].select(range(min(args.max_train_samples, len(dataset["train"]))))
    if args.max_validation_samples is not None:
        dataset["validation"] = dataset["validation"].select(range(min(args.max_validation_samples, len(dataset["validation"]))))
    if args.max_test_samples is not None:
        dataset["test"] = dataset["test"].select(range(min(args.max_test_samples, len(dataset["test"]))))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False, cache_dir=str(HF_HOME / "transformers"))

    def tokenize(batch: dict) -> dict:
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    tokenized = dataset.map(tokenize, batched=True)
    tokenized = tokenized.remove_columns([column for column in tokenized["train"].column_names if column not in {"input_ids", "attention_mask", "label"}])

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label={0: "benign", 1: "unsafe"},
        label2id={"benign": 0, "unsafe": 1},
        cache_dir=str(HF_HOME / "transformers"),
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")

    def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict:
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"],
            "f1": f1_metric.compute(predictions=predictions, references=labels)["f1"],
            "precision": precision_metric.compute(predictions=predictions, references=labels)["precision"],
            "recall": recall_metric.compute(predictions=predictions, references=labels)["recall"],
        }

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        optim=args.optim,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none",
        dataloader_num_workers=0,
        use_cpu=not __import__("torch").cuda.is_available(),
        logging_nan_inf_filter=False,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_pin_memory=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_metrics = trainer.evaluate(tokenized["test"])
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)

    metrics_path = args.output_dir / "test_metrics.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(eval_metrics, handle, indent=2)

    print(json.dumps(eval_metrics, indent=2))


if __name__ == "__main__":
    main()
