from fastapi import APIRouter, Depends

from moderation_service.api.schemas import (
    ModerateRequest,
    ModerateResponse,
    ModerateBatchRequest,
    ModerateBatchResponse,
)
from moderation_service.api.dependencies import get_router
from moderation_service.api.mappers import to_response
from moderation_service.router.hybrid_router import HybridRouter

router = APIRouter(prefix="/moderate", tags=["moderation"])


@router.post("/", response_model=ModerateResponse)
async def moderate(
    request: ModerateRequest,
    hybrid_router: HybridRouter = Depends(get_router),
):
    result = hybrid_router.route(request.text)

    return to_response(request.text, result)


@router.post("/batch", response_model=ModerateBatchResponse)
async def moderate_batch(
    request: ModerateBatchRequest,
    hybrid_router: HybridRouter = Depends(get_router),
):
    results = []

    for text in request.texts:
        result = hybrid_router.route(text)

        results.append(
            to_response(text, result)
        )

    return ModerateBatchResponse(results=results)