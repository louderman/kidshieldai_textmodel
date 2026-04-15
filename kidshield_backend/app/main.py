from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import health, classify, classify_image
from app.core import model as model_module
from app.core import image_model as image_model_module
from app.core.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_module.load_model(settings.model_path)
    image_model_module.load_image_model(settings.vit_model_path)
    yield


app = FastAPI(title="KidShield API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(classify.router)
app.include_router(classify_image.router)
