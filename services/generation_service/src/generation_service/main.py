import logging
import threading

from fastapi import FastAPI

from generation_service.core.logging import setup_logging
from generation_service.api.routes import router
from generation_service.workers.worker import run_worker

setup_logging()

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Generation Service",
    version="1.0.0",
)

app.include_router(router)


@app.on_event("startup")
def startup() -> None:
    thread = threading.Thread(target=run_worker, daemon=True, name="generation-worker")
    thread.start()
    logger.info("Generation worker started")


@app.get("/health")
def health():
    return {"status": "ok"}
