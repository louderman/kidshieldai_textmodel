from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
HF_HOME = ROOT / ".cache" / "huggingface"
HF_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("HF_DATASETS_CACHE", str(HF_HOME / "datasets"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_HOME / "transformers"))


def _default_model_dir() -> Path:
    return Path(os.getenv("MODEL_DIR", str(ROOT / "outputs" / "mdeberta-kidshield")))


@dataclass
class ModelState:
    model_dir: Path
    model_name: str
    tokenizer: AutoTokenizer
    model: AutoModelForSequenceClassification
    device: torch.device

class PredictItem(BaseModel):
    text: str = Field(..., min_length=1, description="Content, URL, or domain to classify.")
    id: str | None = Field(default=None, description="Optional client-provided identifier.")
    kind: Literal["text", "url", "domain"] | None = Field(
        default=None,
        description="Optional hint about the input type.",
    )


class PredictRequest(BaseModel):
    text: str | None = Field(default=None, description="Single input to classify.")
    items: list[PredictItem] | None = Field(default=None, description="Batch inputs.")

    @model_validator(mode="after")
    def _validate_payload(self) -> "PredictRequest":
        if self.text is None and not self.items:
            raise ValueError("Provide either 'text' or 'items' in the request body.")
        if self.text is not None and self.items:
            raise ValueError("Provide only one of 'text' or 'items', not both.")
        return self

class PredictScore(BaseModel):
    benign: float
    unsafe: float

class PredictResponseItem(BaseModel):
    text: str
    label: Literal["benign", "unsafe"]
    is_unsafe: bool
    scores: PredictScore
    id: str | None = None
    kind: Literal["text", "url", "domain"] | None = None

class PredictResponse(BaseModel):
    model: dict
    threshold: float
    latency_ms: int
    items: list[PredictResponseItem]

def _load_model(model_dir: Path) -> ModelState:
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return ModelState(
        model_dir=model_dir,
        model_name=model.config.name_or_path,
        tokenizer=tokenizer,
        model=model,
        device=device,
    )


def create_app() -> FastAPI:
    app = FastAPI(title="Kidshield AI Text Model API", version="0.1.0")
    state: dict[str, ModelState] = {}

    def get_state() -> ModelState:
        if "model" not in state:
            state["model"] = _load_model(_default_model_dir())
        return state["model"]

    @app.get("/health")
    def health() -> dict:
        model_dir = _default_model_dir()
        loaded = "model" in state
        return {
            "status": "ok",
            "model_dir": str(model_dir),
            "model_loaded": loaded,
        }

    @app.get("/info")
    def info() -> dict:
        try:
            model_state = get_state()
        except FileNotFoundError:
            raise HTTPException(status_code=503, detail="Model directory not found.")
        return {
            "model_name": model_state.model_name,
            "model_dir": str(model_state.model_dir),
            "device": str(model_state.device),
            "labels": model_state.model.config.id2label,
        }

    @app.post("/predict", response_model=PredictResponse)
    def predict(payload: PredictRequest) -> PredictResponse:
        try:
            model_state = get_state()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        if payload.text is not None:
            items = [PredictItem(text=payload.text)]
        else:
            items = payload.items or []

        max_batch = int(os.getenv("MAX_BATCH_SIZE", "32"))
        if len(items) > max_batch:
            raise HTTPException(status_code=413, detail=f"Batch too large. Max {max_batch} items.")

        max_length = int(os.getenv("MAX_LENGTH", "256"))
        texts = [item.text for item in items]

        start = time.perf_counter()
        encoded = model_state.tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(model_state.device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = model_state.model(**encoded).logits
            probs = torch.softmax(logits, dim=-1).cpu().tolist()
        latency_ms = int((time.perf_counter() - start) * 1000)
        threshold = float(os.getenv("UNSAFE_THRESHOLD", "0.5"))
        response_items: list[PredictResponseItem] = []
        for item, score_pair in zip(items, probs, strict=True):
            benign_score, unsafe_score = score_pair
            label = "unsafe" if unsafe_score >= threshold else "benign"
            response_items.append(
                PredictResponseItem(
                    text=item.text,
                    id=item.id,
                    kind=item.kind,
                    label=label,
                    is_unsafe=label == "unsafe",
                    scores=PredictScore(benign=benign_score, unsafe=unsafe_score),
                )
            )

        return PredictResponse(
            model={"name": model_state.model_name, "dir": str(model_state.model_dir)},
            threshold=threshold,
            latency_ms=latency_ms,
            items=response_items,
        )

    return app


app = create_app()
