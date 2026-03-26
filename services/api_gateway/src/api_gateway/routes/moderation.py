from api_gateway.clients.moderation_client import ModerationClient
from api_gateway.dependencies import get_moderation_client
from api_gateway.schemas.moderation import (
    ModerateBatchRequest,
    ModerateBatchResponse,
    ModerateRequest,
    ModerateResponse,
)
from fastapi import APIRouter, Depends

router = APIRouter(prefix="/moderation", tags=["moderation"])


@router.post("/", response_model=ModerateResponse)
async def moderate(
    request: ModerateRequest,
    client: ModerationClient = Depends(get_moderation_client),
):
    return await client.moderate(request.text)


@router.post("/batch", response_model=ModerateBatchResponse)
async def moderate_batch(
    request: ModerateBatchRequest,
    client: ModerationClient = Depends(get_moderation_client),
):
    return await client.moderate_batch(request.texts)
