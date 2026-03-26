from pydantic import BaseModel, Field
from typing import List, Optional


class ModerateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)


class ModerateResponse(BaseModel):
    text: str
    is_toxic: bool
    score: float
    reason: Optional[str]
    source: str


class ModerateBatchRequest(BaseModel):
    texts: List[str]


class ModerateBatchResponse(BaseModel):
    results: List[ModerateResponse]