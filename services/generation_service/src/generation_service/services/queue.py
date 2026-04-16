import queue
import threading
import uuid
from dataclasses import dataclass
from typing import Any

from generation_service.core.config import settings
from schemas.generation import GenerateAcceptedResponse, GenerationResultResponse


class QueueFullError(Exception):
    pass


@dataclass(frozen=True)
class GenerationTask:
    task_id: str
    payload: dict[str, Any]


class GenerationTaskQueue:
    def __init__(self, maxsize: int = 0):
        self._queue: queue.Queue[GenerationTask] = queue.Queue(maxsize=maxsize)
        self._results: dict[str, GenerationResultResponse] = {}
        self._lock = threading.Lock()

    def submit(self, payload: dict[str, Any]) -> GenerateAcceptedResponse:
        task_id = str(uuid.uuid4())
        task = GenerationTask(task_id=task_id, payload=payload)

        with self._lock:
            self._results[task_id] = GenerationResultResponse(
                task_id=task_id,
                status="pending",
            )

        try:
            self._queue.put_nowait(task)
        except queue.Full as exc:
            with self._lock:
                self._results.pop(task_id, None)
            raise QueueFullError("Generation queue is full. Try again later.") from exc

        return GenerateAcceptedResponse(task_id=task_id)

    def next_task(self) -> GenerationTask:
        task = self._queue.get()
        self.mark_processing(task.task_id)
        return task

    def mark_processing(self, task_id: str) -> None:
        with self._lock:
            result = self._results.get(task_id)
            if result is None:
                return
            self._results[task_id] = result.model_copy(update={"status": "processing"})

    def mark_done(self, task_id: str, image_path: str) -> None:
        with self._lock:
            self._results[task_id] = GenerationResultResponse(
                task_id=task_id,
                status="done",
                image_path=image_path,
            )

    def mark_failed(self, task_id: str, error: str) -> None:
        with self._lock:
            self._results[task_id] = GenerationResultResponse(
                task_id=task_id,
                status="error",
                error=error,
            )

    def get_result(self, task_id: str) -> GenerationResultResponse | None:
        with self._lock:
            result = self._results.get(task_id)
            return None if result is None else result.model_copy()


task_queue = GenerationTaskQueue(maxsize=settings.MAX_QUEUE_SIZE)
