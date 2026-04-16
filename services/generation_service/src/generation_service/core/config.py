from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_FILE = Path(__file__).resolve().parents[3] / ".env"
SERVICE_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    SERVICE_NAME: str = "generation_service"
    PORT: int = 8100

    MODEL_PATH: str
    OUTPUT_DIR: str
    UPLOAD_DIR: str = "uploads"
    SDXL_MODEL_ID: str = "stabilityai/stable-diffusion-xl-base-1.0"
    INSTANTID_REPO_ID: str = "InstantX/InstantID"
    INSTANTID_CACHE_DIR: str = "models/instantid"
    INSTANTID_FACE_ANALYSIS_ROOT: str = "models/instantid/insightface"
    INSTANTID_STRUCTURE_CONTROLNET_ID: str = "diffusers/controlnet-canny-sdxl-1.0-small"
    INSTANTID_PROVIDERS: str = "CUDAExecutionProvider,CPUExecutionProvider"
    DEVICE: str = "cuda"
    PRECISION: str = "auto"
    MAX_QUEUE_SIZE: int = 0

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()


def resolve_service_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return SERVICE_ROOT / path
