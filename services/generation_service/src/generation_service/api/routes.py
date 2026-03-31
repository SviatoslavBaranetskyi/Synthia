from fastapi import APIRouter, Depends, HTTPException

from generation_service.api.dependencies import get_task_queue
from generation_service.services.queue import GenerationTaskQueue
from schemas.generation import (
    GenerateAcceptedResponse,
    GenerateRequest,
    GenerationResultResponse,
)

router = APIRouter(prefix="/generation", tags=["generation"])


@router.post("/", response_model=GenerateAcceptedResponse)
def generate(
    request: GenerateRequest,
    queue: GenerationTaskQueue = Depends(get_task_queue),
):
    return queue.submit(request.model_dump())


@router.get("/tasks/{task_id}", response_model=GenerationResultResponse)
def get_generation_result(
    task_id: str,
    queue: GenerationTaskQueue = Depends(get_task_queue),
):
    result = queue.get_result(task_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Generation task not found")

    return result
