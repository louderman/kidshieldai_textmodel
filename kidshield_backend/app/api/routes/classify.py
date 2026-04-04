from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.core import model as model_module

router = APIRouter()


class ClassifyRequest(BaseModel):
    text: str


class ClassifyResponse(BaseModel):
    label: str
    score: float


@router.post("/classify", response_model=ClassifyResponse)
def classify(body: ClassifyRequest):
    if not body.text.strip():
        raise HTTPException(status_code=422, detail="text must not be empty")
    result = model_module.predict(body.text)
    return result
