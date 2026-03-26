from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    MODERATION_SERVICE_URL: str
    API_ENV: str = "dev"
    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent.parent.parent / ".env", env_file_encoding="utf-8"
    )


settings = Settings()
