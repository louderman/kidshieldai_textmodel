# Kidshield AI DeBERTa Fine-Tuning

This workspace prepares and fine-tunes a single `microsoft/mdeberta-v3-base` classifier across:

- Jigsaw Toxic Comment Classification Challenge
- Jigsaw Multilingual Toxic Comment Classification
- HateXplain
- Phishing.Database
- Majestic Million

All sources are merged into one binary target:

- `label = 1`: unsafe content or phishing indicators
- `label = 0`: benign content or benign domains

`mdeberta-v3-base` is used instead of English-only DeBERTa because the dataset mix includes multilingual text.

## Layout

```text
data/
  raw/
    jigsaw-toxic/
    jigsaw-multilingual/
    hatexplain/
    phishing-database/
    majestic-million/
  processed/
outputs/
scripts/
src/kidshield_ai/
```

## Python Environment

Use Python `3.11` for training. Python `3.14` caused compatibility issues with older `tokenizers` wheels and the more stable `transformers` stack for DeBERTa.

Recommended environment:

```bash
python3.11 -m venv .venv311
.venv311/bin/python -m pip install --upgrade pip
.venv311/bin/python -m pip install \
  "torch==2.10.0" \
  "transformers==4.46.3" \
  "tokenizers<0.21" \
  "accelerate<1.0" \
  datasets evaluate pandas scikit-learn sentencepiece protobuf tiktoken
```

Quick verification:

```bash
.venv311/bin/python -c "import torch, transformers; print(torch.__version__, torch.cuda.is_available(), transformers.__version__)"
```

## Raw Data

Expected raw files:

- `data/raw/jigsaw-toxic/train.csv`
- `data/raw/jigsaw-multilingual/validation.csv`
- `data/raw/hatexplain/dataset.json`
- `data/raw/hatexplain/post_id_divisions.json`
- `data/raw/phishing-database/phishing-links-ACTIVE.txt`
- `data/raw/phishing-database/phishing-domains-ACTIVE.txt`
- `data/raw/majestic-million/majestic_million.csv`

Dataset mapping:

- Jigsaw Toxic contributes English toxic and non-toxic comments.
- Jigsaw Multilingual contributes multilingual toxic and non-toxic comments from `validation.csv`.
- HateXplain maps `hatespeech` and `offensive` to positive, `normal` to negative.
- Phishing.Database contributes phishing URLs and domains as positive.
- Majestic Million contributes benign domains as negative.

## Downloading Datasets

Public-source downloads can be automated with:

```bash
.venv311/bin/python scripts/download_datasets.py --raw-dir data/raw
```

This downloads:

- Jigsaw Toxic from a public mirror
- HateXplain from GitHub
- Phishing.Database from GitHub
- Majestic Million from a GitHub mirror

The multilingual Jigsaw competition file still has to be downloaded manually from Kaggle.

Kaggle multilingual dataset page:

- https://www.kaggle.com/competitions/jigsaw-multilingual-toxic-comment-classification

Manual multilingual step:

1. Join the competition on Kaggle.
2. Download `jigsaw-multilingual-toxic-comment-classification.zip`.
3. Extract `validation.csv`.
4. Place it at `data/raw/jigsaw-multilingual/validation.csv`.

Example extraction:

```bash
unzip -j -o ~/Downloads/jigsaw-multilingual-toxic-comment-classification.zip validation.csv -d data/raw/jigsaw-multilingual
```

## Preprocessing

Build the merged binary dataset with:

```bash
.venv311/bin/python scripts/prepare_datasets.py \
  --raw-dir data/raw \
  --output-dir data/processed \
  --majestic-limit 200000 \
  --phishing-limit 200000
```

Outputs:

- `data/processed/train.jsonl`
- `data/processed/validation.jsonl`
- `data/processed/test.jsonl`
- `data/processed/dataset_report.json`

The JSONL files intentionally serialize only:

- `text`
- `label`
- `source`

This avoids schema conflicts from nested metadata fields across different datasets.

## Training

The trainer stores Hugging Face caches under `.cache/huggingface` inside the workspace so it does not depend on a writable home-directory cache.

### Smoke Test

Run this first to verify numerical stability and memory usage:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv311/bin/python scripts/train.py \
  --data-dir data/processed \
  --output-dir outputs/mdeberta-kidshield-smoke \
  --model-name microsoft/mdeberta-v3-base \
  --epochs 1 \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --lr 5e-6 \
  --fp16 \
  --gradient-checkpointing \
  --optim adamw_torch \
  --max-train-samples 512 \
  --max-validation-samples 128 \
  --max-test-samples 128
```

What to look for:

- `loss` should be finite
- `grad_norm` should be finite
- no CUDA OOM
- `eval_loss` should be finite

Accuracy by itself is not enough on the small smoke split because the labels are imbalanced.

### Full Training

For an 8 GB GPU, start with:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv311/bin/python scripts/train.py \
  --data-dir data/processed \
  --output-dir outputs/mdeberta-kidshield \
  --model-name microsoft/mdeberta-v3-base \
  --epochs 3 \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --lr 1e-5 \
  --fp16 \
  --gradient-checkpointing \
  --optim adamw_torch
```

This uses:

- micro-batch size `1`
- gradient accumulation `8`
- `fp16`
- gradient checkpointing

That combination is much more practical on a 7.5-8 GB GPU than a naive batch size of `8`.

## Known Warnings

These are expected:

- `Some weights ... were not initialized ... classifier/pooler`
  - normal when loading a base checkpoint into a sequence-classification head
- `UNEXPECTED` and `MISSING` keys for the task head
  - expected for checkpoint-to-task adaptation

Warnings that are not acceptable:

- `loss: nan`
- `grad_norm: nan`
- `eval_loss: nan`
- repeated all-negative predictions with `eval_f1 = 0`

If those appear in a smoke run, stop and debug before launching full training.

## Current Caveats

- The pipeline trains a single binary classifier, not separate heads for toxicity subtype prediction.
- The current trainer does not yet apply class-weighted loss, so full training can still bias toward the negative class. If metrics collapse to all-negative predictions, add weighted cross-entropy next.
- URL/domain classification would likely benefit from a character-aware or byte-level model, but this setup keeps one shared text encoder for operational simplicity.
