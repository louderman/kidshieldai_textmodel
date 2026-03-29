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

### Larger Smoke Test

This is the first smoke configuration that produced a useful non-zero F1 signal:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv311/bin/python scripts/train.py \
  --data-dir data/processed \
  --output-dir outputs/mdeberta-kidshield-smoke-large \
  --model-name microsoft/mdeberta-v3-base \
  --epochs 1 \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --lr 5e-6 \
  --fp16 \
  --gradient-checkpointing \
  --optim adamw_torch \
  --class-weighting \
  --max-train-samples 4096 \
  --max-validation-samples 512 \
  --max-test-samples 512
```

This took about 6 minutes locally and produced a non-zero eval F1, which made it a useful go/no-go check before full training.

## Cloud Training

If local training is too slow, use a cloud GPU. A 24 GB or 48 GB card is much more practical than an 8 GB desktop GPU for `mdeberta-v3-base`.

Suggested workflow:

1. Push this repository to GitHub.
2. Launch a cloud GPU instance.
3. Clone the repo on the remote machine.
4. Run the setup script.
5. Upload or download the datasets.
6. Preprocess if needed.
7. Run training.

Recommended providers:

- Lambda GPU Cloud
- Runpod

On a cloud box with more VRAM, you can usually relax memory-saving settings and train faster.

### Cloud Setup

After SSHing into the instance:

```bash
git clone <your-repo-url>
cd "Kidshield AI"
bash scripts/setup_cloud.sh
```

If the remote machine uses a different Python binary:

```bash
PYTHON_BIN=python3.12 bash scripts/setup_cloud.sh
```

### Getting Data Onto The Cloud Box

Options:

- copy `data/raw/` or `data/processed/` from your local machine with `rsync` or `scp`
- redownload the public datasets on the remote box
- manually upload the Kaggle multilingual `validation.csv`

If you already have processed files locally, copying `data/processed/` is usually the fastest path.

Example from your local machine:

```bash
rsync -avz data/processed/ <user>@<host>:"/path/to/Kidshield AI/data/processed/"
```

If you want to copy raw data too:

```bash
rsync -avz data/raw/ <user>@<host>:"/path/to/Kidshield AI/data/raw/"
```

### Cloud Training Command

Use the helper script:

```bash
bash scripts/train_cloud.sh
```

Defaults in `scripts/train_cloud.sh`:

- model: `microsoft/mdeberta-v3-base`
- epochs: `3`
- batch size: `4`
- grad accumulation: `1`
- lr: `1e-5`
- `fp16`
- class weighting enabled

Override values with env vars:

```bash
OUTPUT_DIR=outputs/mdeberta-kidshield-a10 \
BATCH_SIZE=8 \
GRAD_ACCUM=1 \
EPOCHS=3 \
LR=1e-5 \
bash scripts/train_cloud.sh
```

If you want to bypass the helper:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv311/bin/python scripts/train.py \
  --data-dir data/processed \
  --output-dir outputs/mdeberta-kidshield-cloud \
  --model-name microsoft/mdeberta-v3-base \
  --epochs 3 \
  --batch-size 4 \
  --lr 1e-5 \
  --fp16 \
  --optim adamw_torch \
  --class-weighting
```

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
- The trainer now supports `--class-weighting`, which improved larger smoke-test behavior, but recall is still likely to need tuning.
- URL/domain classification would likely benefit from a character-aware or byte-level model, but this setup keeps one shared text encoder for operational simplicity.
