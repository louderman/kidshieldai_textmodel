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

## Inference API

A FastAPI backend is included for serving model predictions from a trained checkpoint.

**Working directory:** run every command in this section from the **repository root** (`kidshieldai_textmodel/`), the same folder that contains `requirements.txt` and `scripts/`. If you run `pip install` or `python scripts/serve_api.py` from the parent `Project_Work` folder, paths will not resolve.

### Install API Dependencies

```bash
pip install -r requirements.txt
```

### Run the Server

```bash
python scripts/serve_api.py
```

By default the server runs at:

- `http://localhost:8000`

### Configure Runtime

Optional environment variables:

- `MODEL_DIR` (default: `outputs/mdeberta-kidshield`)
- `HOST` (default: `0.0.0.0`)
- `PORT` (default: `8000`)
- `MAX_LENGTH` (default: `256`)
- `MAX_BATCH_SIZE` (default: `32`)
- `UNSAFE_THRESHOLD` (default: `0.5`)

### Team integration and handoff

Use this section when one person trains the model and another runs or embeds the HTTP API.

**Single model, one decision.** Training merges toxicity / hate-related text (Jigsaw, HateXplain, etc.) and phishing URLs or domains (plus benign domains) into **one binary classifier**: `benign` vs `unsafe`. There is **not** a second model or separate HTTP route for “URL vs hate”; every input is a string passed through the same encoder. Optional `kind` in JSON (`text`, `url`, `domain`) is **metadata for clients only** and does not switch models.

**Checkpoint location (`MODEL_DIR`).** After `scripts/train.py` finishes, `trainer.save_model()` and `tokenizer.save_pretrained()` write a directory that Hugging Face can reload. Point the API at that folder:

- Default relative path: `outputs/mdeberta-kidshield` under the repo root.
- Another machine or teammate checkout: set an **absolute** path, for example:

```bash
export MODEL_DIR="/absolute/path/to/trained-checkpoint"
python scripts/serve_api.py
```

That directory should contain the usual Hugging Face artifacts (for example `config.json`, tokenizer files, and model weights). If the path is missing, `/predict` and `/info` return **503** until a valid `MODEL_DIR` exists.

**Inference does not modify the checkpoint.** The API loads weights read-only, runs `model.eval()`, and uses `torch.no_grad()` for forward passes. It does **not** fine-tune or overwrite files under `MODEL_DIR`. Training remains a separate step (`scripts/train.py`).

**Git and large files.** Trained weights and caches are **not** meant to be committed: `.gitignore` includes `outputs/` and `.cache/`. The handoff is usually “share the API code via git” and “share the checkpoint via drive, artifact store, or agreed folder,” then set `MODEL_DIR` accordingly.

**Environment alignment.** For consistent behavior with training, use the same **Python version** and compatible **PyTorch / transformers** stack as documented in [Python Environment](#python-environment) (this project targets Python `3.11` for training).

**Base URL for downstream services.** Other components should call:

- `http://<HOST>:<PORT>` with `HOST`/`PORT` from the environment (defaults `0.0.0.0:8000`; for local-only clients on the same machine, you can set `HOST=127.0.0.1`).

**Integration contract (minimal).**

| Concern | Detail |
| --- | --- |
| Classify one string | `POST /predict` with body `{"text":"<string>"}` |
| Classify many | `POST /predict` with body `{"items":[{"text":"..."}, ...]}` |
| Labels | Response `label` is `"benign"` or `"unsafe"`; `scores` gives softmax probabilities for index order `benign` then `unsafe` |
| Health | `GET /health` for process up; first `/predict` or `/info` triggers model load |

**Example payloads by input type** (all use the same endpoint and model):

Toxic or hateful **comment** (natural language):

```json
{
  "text": "Example of targeted harassment or slurs goes here."
}
```

**URL** string (phishing-style link as a single field):

```json
{
  "text": "http://totally-legit-bank.example/verify-account"
}
```

**Domain** only (no scheme):

```json
{
  "text": "suspicious-phishing.example"
}
```

Batch mixing types (optional `id` / `kind` for your app’s logging):

```json
{
  "items": [
    { "id": "c1", "kind": "text", "text": "Hostile message text here." },
    { "id": "u1", "kind": "url", "text": "https://evil.example/login" },
    { "id": "d1", "kind": "domain", "text": "benign-site.com" }
  ]
}
```

**Interactive API docs.** With the server running, open `http://localhost:8000/docs` (Swagger UI) or `http://localhost:8000/redoc` for the live schema and “try it” forms.

### Endpoints

- `GET /health`: service heartbeat and model-load status
- `GET /info`: model metadata (name, labels, device)
- `POST /predict`: classify one input or a batch

### `POST /predict` Request Payload

Use either a single `text` field:

```json
{
  "text": "Free money!!! Click this link now"
}
```

Or a batch `items` array:

```json
{
  "items": [
    {
      "id": "msg-1",
      "text": "You won a prize. Claim now.",
      "kind": "text"
    },
    {
      "id": "dom-2",
      "text": "openai.com",
      "kind": "domain"
    }
  ]
}
```

Notes:

- Provide only one of `text` or `items`
- `kind` is optional (`text`, `url`, `domain`) and is a hint only

### `POST /predict` Response Shape

```json
{
  "model": {
    "name": "microsoft/mdeberta-v3-base",
    "dir": "outputs/mdeberta-kidshield"
  },
  "threshold": 0.5,
  "latency_ms": 23,
  "items": [
    {
      "id": "msg-1",
      "kind": "text",
      "text": "You won a prize. Claim now.",
      "label": "unsafe",
      "is_unsafe": true,
      "scores": {
        "benign": 0.08,
        "unsafe": 0.92
      }
    }
  ]
}
```

### Quick `curl` Examples

Health check:

```bash
curl -s http://localhost:8000/health
```

Single prediction:

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"This is a phishing message. Verify your account now."}'
```

Batch prediction:

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {"id":"1","text":"Click this suspicious link now","kind":"text"},
      {"id":"2","text":"google.com","kind":"domain"}
    ]
  }'
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
- The current trainer does not yet apply class-weighted loss, so full training can still bias toward the negative class. If metrics collapse to all-negative predictions, add weighted cross-entropy next.
- URL/domain classification would likely benefit from a character-aware or byte-level model, but this setup keeps one shared text encoder for operational simplicity.
