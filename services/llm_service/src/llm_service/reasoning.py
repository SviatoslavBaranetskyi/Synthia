from typing import List


def build_prompt(text: str, examples: List[str]) -> str:
    examples_str = "\n".join([f"- {ex}" for ex in examples])

    return f"""
        You are a strict content moderation system.

        Your task is to detect ANY form of toxicity, including:
        - insults
        - passive-aggressive language
        - sarcasm targeting a person
        - condescending or demeaning tone

        Even subtle or indirect insults MUST be classified as "toxic".

        Text:
        \"{text}\"

        Similar examples:
        {examples_str}

        Respond ONLY with valid JSON:

        {{
        "label": "toxic" or "safe",
        "confidence": float (0 to 1)
        }}
    """
