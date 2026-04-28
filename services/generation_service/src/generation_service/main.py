from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import logging
import threading

from fastapi import FastAPI

from generation_service.core.logging import setup_logging
from generation_service.api.routes import image_router, router
from generation_service.core.config import settings
from generation_service.services.image_generation import image_generation_service
from generation_service.workers.worker import run_worker

setup_logging()

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    thread = threading.Thread(target=run_worker, daemon=True, name="generation-worker")
    thread.start()
    logger.info("Generation worker started")

    if settings.ZIMAGE_ENABLED:
        await image_generation_service.startup()

    try:
        yield
    finally:
        if settings.ZIMAGE_ENABLED:
            await image_generation_service.shutdown()


app = FastAPI(
    title="Generation Service",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)
app.include_router(image_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
