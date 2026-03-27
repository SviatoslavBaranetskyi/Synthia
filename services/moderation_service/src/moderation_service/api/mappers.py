from schemas.moderation import ModerateResponse
from moderation_service.router.schemas import ModerationResult


def to_response(text: str, result: ModerationResult) -> ModerateResponse:
    return ModerateResponse(
        text=text,
        is_toxic=result.label == "toxic",
        score=result.probability or result.llm_confidence or 0.0,
        reason=result.reason,
        source=result.source,
    )
