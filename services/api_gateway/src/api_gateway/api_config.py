from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    MODERATION_SERVICE_URL: str
    GENERATION_SERVICE_URL: str
    API_ENV: str = "dev"
    LOG_LEVEL: str = "INFO"
    CORS_ALLOW_ORIGINS: str = ""

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
    )

    @property
    def cors_allow_origins(self) -> list[str]:
        return [
            origin.strip()
            for origin in self.CORS_ALLOW_ORIGINS.split(",")
            if origin.strip()
        ]


settings = Settings()  # type: ignore[call-arg]
