from dataclasses import dataclass


@dataclass
class SimilarExample:
    text: str
    score: float