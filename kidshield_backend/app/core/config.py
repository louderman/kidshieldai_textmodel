from pathlib import Path
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    app_name: str = "KidShield API"
    debug: bool = False
    host: str = "127.0.0.1"
    port: int = 8000
    model_path: str = str(BASE_DIR / "models" / "mdeberta-kidshield")

settings = Settings()
