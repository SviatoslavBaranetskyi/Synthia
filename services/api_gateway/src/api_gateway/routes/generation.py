from api_gateway.clients.generation_client import GenerationClient
from api_gateway.dependencies import get_generation_client
from fastapi import APIRouter, Depends
from schemas.generation import (
    GenerateAcceptedResponse,
    GenerateRequest,
    GenerationResultResponse,
)

router = APIRouter(prefix="/generation", tags=["generation"])


@router.post("/", response_model=GenerateAcceptedResponse)
async def generate(
    request: GenerateRequest,
    client: GenerationClient = Depends(get_generation_client),
):
    return await client.generate(request)


@router.get("/tasks/{task_id}", response_model=GenerationResultResponse)
async def get_generation_result(
    task_id: str,
    client: GenerationClient = Depends(get_generation_client),
):
    return await client.get_result(task_id)
