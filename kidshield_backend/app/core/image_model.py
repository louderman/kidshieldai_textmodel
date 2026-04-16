from __future__ import annotations

import base64
import io
import logging
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)

_image_model = None

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 4-class model: explicit=0, safe_nsfw=1, violent=2, safe_rwf=3
_EXPLICIT_THRESHOLD = 0.45
_VIOLENT_THRESHOLD  = 0.55


def load_image_model(model_path: str) -> None:
    global _image_model

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"ViT model not found at '{path}'. "
            "Make sure kidshield_vit_scripted.pt is in models/vit/"
        )

    logger.info("Loading image model from %s", path)
    _image_model = torch.jit.load(str(path), map_location="cpu")
    _image_model.eval()
    logger.info("Image model loaded.")


def predict_image(image_b64: str) -> dict:
    if _image_model is None:
        raise RuntimeError("Image model is not loaded.")

    image_bytes = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = _transform(image).unsqueeze(0)

    with torch.no_grad():
        logits = _image_model(tensor)

    probs = torch.softmax(logits, dim=-1).squeeze()

    explicit_prob = float(probs[0])
    violent_prob  = float(probs[2])



    if (explicit_prob >= _EXPLICIT_THRESHOLD or
            violent_prob >= _VIOLENT_THRESHOLD):
        predicted_id = int(torch.argmax(probs))
        score = float(probs[predicted_id])
        return {"label": "unsafe", "score": round(score, 4)}

    return {"label": "benign", "score": round(float(probs[3]), 4)}
