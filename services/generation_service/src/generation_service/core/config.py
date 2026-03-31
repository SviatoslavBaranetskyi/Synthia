from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_FILE = Path(__file__).resolve().parents[3] / ".env"


class Settings(BaseSettings):
    SERVICE_NAME: str = "generation_service"
    PORT: int = 8100

    MODEL_PATH: str
    OUTPUT_DIR: str
    DEVICE: str = "cuda"
    PRECISION: str = "auto"
    MAX_QUEUE_SIZE: int = 0

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
