import logging

from api_gateway.api_config import settings
from api_gateway.routes import generation, health, moderation
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

cors_allow_origins = settings.cors_allow_origins

app = FastAPI(
    title="Synthia API Gateway",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_allow_origins,
    allow_credentials=bool(cors_allow_origins),
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(moderation.router)
app.include_router(generation.router)
app.include_router(generation.image_router)
