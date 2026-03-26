import json
import logging
import re
import time
from typing import Any, Dict, Optional

import litellm
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class LLMResponse(BaseModel):
    label: str
    confidence: float
    reason: Optional[str] = None


class LLMClient:
    def __init__(
        self,
        model: str,
        timeout: int = 10,
        max_retries: int = 3,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.api_base = api_base
        self.api_key = api_key

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
        match = re.search(r"\{.*\}", text, re.DOTALL)

        if not match:
            raise ValueError("No JSON found")

        return match.group(0)

    def _parse_response(self, text: str) -> Dict[str, Any]:
        try:
            json_str = self._extract_json(text)
            data = json.loads(json_str)

            parsed = LLMResponse(**data)

            result = parsed.dict()

            if not result.get("reason"):
                result["reason"] = self._fallback_reason(result["label"])

            return result

        except Exception:
            logger.error(f"Invalid LLM response: {text}")

            return {
                "label": "safe",
                "confidence": 0.0,
                "reason": "llm: parse failed",
            }

    def complete(self, prompt: str) -> Dict[str, Any]:
        for attempt in range(self.max_retries):
            try:
                raw = self._call_llm(prompt)
                result = self._parse_response(raw)

                if "reason" not in result:
                    result["reason"] = "llm: no reason provided"

                return result

            except Exception as e:
                wait = 2**attempt
                logger.warning(
                    f"[LLM] retry {attempt + 1}/{self.max_retries} after error: {e}"
                )
                time.sleep(wait)

        logger.error("[LLM] failed after retries, using fallback")

        return {
            "label": "safe",
            "confidence": 0.0,
            "reason": "llm: fallback triggered due to error",
        }

    def _fallback_reason(self, label: str) -> str:
        if label == "toxic":
            return "llm: classified as toxic (no explanation provided)"
        return "llm: classified as safe (no issues detected)"
