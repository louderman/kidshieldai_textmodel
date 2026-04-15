from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.core import image_model as image_model_module

router = APIRouter()


class ClassifyImageRequest(BaseModel):
    image: str  # base64-encoded image bytes


class ClassifyImageResponse(BaseModel):
    label: str
    score: float


@router.post("/classify-image", response_model=ClassifyImageResponse)
def classify_image(body: ClassifyImageRequest):
    if not body.image.strip():
        raise HTTPException(status_code=422, detail="image must not be empty")
    try:
        result = image_model_module.predict_image(body.image)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return result
