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


_BLOCKLIST = {
    "shit", "fuck", "fuk", "fck", "fucker", "fucking", "fucked",
    "bitch", "bitches", "ass", "asshole", "bastard",
    "dick", "cock", "pussy", "cunt",
    "nigger", "nigga", "faggot", "fag",
    "whore", "slut", "rape", "rapist",
    "porn", "sex", "nude", "naked",
    "kill", "murder", "suicide",
}

def _blocklist_hit(text: str) -> bool:
    words = re.findall(r'[a-z]+', text.lower())
    return any(w in _BLOCKLIST for w in words)


_SUSPICIOUS_TLDS = {
    ".ru", ".tk", ".gq", ".ml", ".cf", ".pw",
    ".xyz", ".top", ".click", ".link", ".online",
    ".site", ".club", ".info", ".biz",
}

_BRAND_NAMES = {
    "paypal", "apple", "microsoft", "google", "amazon",
    "netflix", "instagram", "facebook", "steam", "roblox",
    "twitter", "tiktok", "snapchat", "discord", "spotify",
    "bankofamerica", "chase", "wellsfargo", "ebay", "walmart",
}

_PHISHING_KEYWORDS = {
    "verify", "secure", "login", "signin", "account", "update",
    "confirm", "billing", "alert", "support", "claim", "free",
    "gift", "suspended", "unusual", "activity", "recover", "unlock",
    "validate", "authenticate", "password", "credential",
}

def _is_suspicious_domain(text: str) -> bool:
    # only apply to inputs that look like hostnames (has dots, no spaces)
    t = text.strip().lower()
    if ' ' in t or '.' not in t:
        return False

    # strip port if present
    t = t.split(':')[0]

    has_suspicious_tld = any(t.endswith(tld) for tld in _SUSPICIOUS_TLDS)
    parts = set(re.findall(r'[a-z]+', t))
    has_brand = bool(parts & _BRAND_NAMES)
    has_phishing_kw = bool(parts & _PHISHING_KEYWORDS)

    # flag if: suspicious TLD, or brand and phishing keyword combo
    return has_suspicious_tld or (has_brand and has_phishing_kw)


def predict(text: str) -> dict:
    if _tokenizer is None or _model is None:
        raise RuntimeError("Model is not loaded.")

    normalized = normalize(text)

    if _blocklist_hit(normalized):
        return {"label": "unsafe", "score": 1.0}

    if _is_suspicious_domain(text):
        return {"label": "unsafe", "score": 1.0}

    inputs = _tokenizer(
        normalized,
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
