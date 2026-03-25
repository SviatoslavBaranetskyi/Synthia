import json
import logging
import time
from typing import Any, Dict, Optional

import litellm
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


class LLMResponse(BaseModel):
    label: str
    confidence: float


class LLMClient:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        timeout: int = 10,
        max_retries: int = 3,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

        if api_base:
            litellm.api_base = api_base

        if api_key:
            litellm.api_key = api_key

    def _call_llm(self, prompt: str) -> str:
        response = litellm.completion(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0.0,
            timeout=self.timeout,
        )

        return response["choices"][0]["message"]["content"]

    def _extract_json(self, text: str) -> str:
        start = text.find("{")
        end = text.rfind("}") + 1

        if start == -1 or end == -1:
            raise ValueError("No JSON found in response")

        return text[start:end]

    def _parse_response(self, text: str) -> Dict[str, Any]:
        try:
            json_str = self._extract_json(text)
            data = json.loads(json_str)

            parsed = LLMResponse(**data)

            return parsed.dict()

        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            logger.error(f"Invalid LLM response: {text}")
            raise ValueError("Invalid LLM response") from e

    def complete(self, prompt: str) -> Dict[str, Any]:
        last_error = None

        for attempt in range(self.max_retries):
            try:
                raw = self._call_llm(prompt)
                return self._parse_response(raw)

            except Exception as e:
                last_error = e
                wait = 2**attempt

                logger.warning(
                    f"[LLM] retry {attempt + 1}/{self.max_retries} after error: {e}"
                )

                time.sleep(wait)

        logger.error("[LLM] failed after retries, using fallback")

        return {
            "label": "safe",
            "confidence": 0.0,
        }
