from dataclasses import dataclass
from typing import List, TypedDict


class ModerationState(TypedDict, total=False):
    text: str

    # ML
    probability: float
    label: int
    reason: str

    # FAISS
    similar_examples: List[str]

    # LLM
    llm_label: str
    llm_confidence: float
    llm_reason: str

    # final
    final_label: str
    source: str


@dataclass
class ModerationResult:
    label: str
    source: str
    probability: float | None = None
    llm_confidence: float | None = None
    reason: str | None = None
