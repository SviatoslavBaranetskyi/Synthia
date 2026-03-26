from typing import List


def build_prompt(text: str, examples: List[str]) -> str:
    examples_str = "\n".join([f"- {ex}" for ex in examples]) or "None"

    return f"""
        You are a content moderation API.

        You MUST return a JSON object.
        DO NOT write anything except JSON.

        STRICT RULES:
        - Classify as "toxic" ONLY if there is clear harmful intent.
        - Harmful intent includes:
        - direct insults ("you are stupid")
        - threats ("I will destroy you")
        - harassment or demeaning language
        - DO NOT classify neutral, friendly, or ambiguous text as toxic.
        - If unsure → return "safe"

        Text:
        \"{text}\"

        Similar examples:
        {examples_str}

        Required JSON schema:
        {{
        "label": "toxic" | "safe",
        "confidence": number between 0 and 1,
        "reason": short explanation (REQUIRED)
        }}

        Guidelines for reason:
        - Be concise (2-6 words)
        - Describe WHY (e.g. "direct insult", "threat", "friendly message")

        Valid examples:
        {{"label":"toxic","confidence":0.92,"reason":"direct insult"}}
        {{"label":"safe","confidence":0.05,"reason":"friendly greeting"}}
    """
