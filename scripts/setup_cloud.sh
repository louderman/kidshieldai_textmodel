#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
VENV_DIR="${VENV_DIR:-.venv311}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Missing Python binary: $PYTHON_BIN" >&2
  exit 1
fi

"$PYTHON_BIN" -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install \
  "torch==2.10.0" \
  "transformers==4.46.3" \
  "tokenizers<0.21" \
  "accelerate<1.0" \
  datasets evaluate pandas scikit-learn sentencepiece protobuf tiktoken

"$VENV_DIR/bin/python" -c "import torch, transformers; print(torch.__version__, torch.cuda.is_available(), transformers.__version__)"
