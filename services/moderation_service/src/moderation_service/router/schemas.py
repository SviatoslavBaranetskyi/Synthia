from typing import TypedDict, List, Optional
from dataclasses import dataclass


class ModerationState(TypedDict, total=False):
    text: str

    # ML
    probability: float
    label: int

    # FAISS
    similar_examples: List[str]

    # LLM
    llm_label: str
    llm_confidence: float

    # final
    final_label: str
    source: str


@dataclass
class ModerationResult:
    label: str
    source: str
    probability: float | None = None
    llm_confidence: float | None = None