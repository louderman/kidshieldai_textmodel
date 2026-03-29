#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${VENV_DIR:-.venv311}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/mdeberta-kidshield-cloud}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LR="${LR:-1e-5}"
MODEL_NAME="${MODEL_NAME:-microsoft/mdeberta-v3-base}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

"$VENV_DIR/bin/python" scripts/train.py \
  --data-dir data/processed \
  --output-dir "$OUTPUT_DIR" \
  --model-name "$MODEL_NAME" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --gradient-accumulation-steps "$GRAD_ACCUM" \
  --lr "$LR" \
  --fp16 \
  --optim adamw_torch \
  --class-weighting
