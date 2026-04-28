from functools import lru_cache

from generation_service.services.image_generation import (
    ImageGenerationService,
    image_generation_service,
)
from generation_service.services.queue import GenerationTaskQueue, task_queue


@lru_cache
def get_task_queue() -> GenerationTaskQueue:
    return task_queue


@lru_cache
def get_image_generation_service() -> ImageGenerationService:
    return image_generation_service
