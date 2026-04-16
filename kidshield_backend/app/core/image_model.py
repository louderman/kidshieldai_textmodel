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

# 4-class model output mapped to binary label
_ID2LABEL = {0: "explicit", 1: "safe_nsfw", 2: "violent", 3: "safe_rwf"}
_UNSAFE_CLASSES = {"explicit", "violent", "safe_nsfw"}


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
    predicted_id = int(torch.argmax(probs))
    raw_label = _ID2LABEL.get(predicted_id, "safe_rwf")
    label = "unsafe" if raw_label in _UNSAFE_CLASSES else "benign"
    score = float(probs[predicted_id])

    return {"label": label, "score": round(score, 4)}
