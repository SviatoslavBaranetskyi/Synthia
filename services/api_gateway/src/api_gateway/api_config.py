from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    MODERATION_SERVICE_URL: str
    GENERATION_SERVICE_URL: str
    API_ENV: str = "dev"
    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
    )


settings = Settings()
