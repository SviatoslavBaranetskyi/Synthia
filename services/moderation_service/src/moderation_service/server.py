import logging

from fastapi import FastAPI
from moderation_service.api.exceptions import ServiceError, service_exception_handler
from moderation_service.api.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

app = FastAPI(
    title="Moderation Service",
    version="1.0.0",
)

app.include_router(router)

app.add_exception_handler(ServiceError, service_exception_handler)


@app.get("/health")
def health():
    return {"status": "ok"}
