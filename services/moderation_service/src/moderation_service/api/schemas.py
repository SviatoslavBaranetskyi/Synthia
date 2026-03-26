from pydantic import BaseModel
from typing import List, Optional


class ModerateRequest(BaseModel):
    text: str


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