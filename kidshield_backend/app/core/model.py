from __future__ import annotations

import logging
import re
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

_tokenizer = None
_model = None


def load_model(model_path: str) -> None:
    global _tokenizer, _model

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at '{path}'. "
            "Copy the trained model from kidshieldai_textmodel/outputs/mdeberta-kidshield/ "
            "into kidshield_backend/models/mdeberta-kidshield/"
        )

    logger.info("Loading model from %s", path)
    _tokenizer = AutoTokenizer.from_pretrained(str(path))
    _model = AutoModelForSequenceClassification.from_pretrained(str(path))
    _model.eval()
    logger.info("Model loaded — labels: %s", _model.config.id2label)



_LEET = {
    '0': 'o', '1': 'i', '3': 'e', '4': 'a',
    '5': 's', '6': 'g', '7': 't', '8': 'b',
    '9': 'g', '@': 'a', '$': 's', '!': 'i',
    '+': 't', '|': 'i',
}
_TOKEN = re.compile(r'\S+')


def _decode_token(match: re.Match) -> str:
    token = match.group()
    # only decode tokens that contain at least one letter (skips standalone numbers)
    if any(c.isalpha() for c in token):
        return ''.join(_LEET.get(c, c) for c in token)
    return token


def normalize(text: str) -> str:
    return _TOKEN.sub(_decode_token, text)


def predict(text: str) -> dict:
    if _tokenizer is None or _model is None:
        raise RuntimeError("Model is not loaded.")

    inputs = _tokenizer(
        normalize(text),
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    with torch.no_grad():
        logits = _model(**inputs).logits

    probs = torch.softmax(logits, dim=-1).squeeze()
    predicted_id = int(torch.argmax(probs))
    label = _model.config.id2label[predicted_id]
    score = float(probs[predicted_id])

    return {"label": label, "score": round(score, 4)}
