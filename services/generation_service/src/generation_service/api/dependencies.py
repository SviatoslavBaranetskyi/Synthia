from functools import lru_cache

from generation_service.services.queue import GenerationTaskQueue, task_queue


@lru_cache
def get_task_queue() -> GenerationTaskQueue:
    return task_queue
