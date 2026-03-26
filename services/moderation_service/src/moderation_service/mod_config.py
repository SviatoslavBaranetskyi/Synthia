from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    SERVICE_NAME: str = "moderation_service"
    PORT: int = 8001

    MODEL_PATH: str
    EMBEDDING_INDEX_PATH: str

    LLM_MODEL: str
    LLM_API_BASE: str | None = None
    LLM_API_KEY: str | None = None

    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent.parent.parent / ".env", env_file_encoding="utf-8"
    )


settings = Settings()
