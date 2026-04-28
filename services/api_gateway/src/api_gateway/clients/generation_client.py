from pathlib import Path
from typing import Any

import httpx
from fastapi import HTTPException, UploadFile

from api_gateway.api_config import settings
from schemas.generation import (
    GenerateImageRequest,
    GenerateRequest,
    GenerationModelKey,
    GenerationSampler,
)


class GenerationClient:
    def __init__(self) -> None:
        self.base_url = settings.GENERATION_SERVICE_URL.rstrip("/")

    async def generate(self, request: GenerateRequest) -> Any:
        return await self._request_json(
            "POST",
            "/generation/",
            timeout=30.0,
            json=request.model_dump(exclude_none=True),
        )

    async def list_models(self) -> Any:
        return await self._request_json(
            "GET",
            "/generation/models",
            timeout=15.0,
        )

    async def generate_image(self, request: GenerateImageRequest) -> Any:
        return await self._request_json(
            "POST",
            "/generate/image",
            timeout=90.0,
            json=request.model_dump(exclude_none=True),
        )

    async def edit_image(
        self,
        image: UploadFile,
        prompt: str,
        negative_prompt: str = "",
        model_key: GenerationModelKey = "sdxl_instantid",
        sampler: GenerationSampler = "dpmpp_sde_karras",
        steps: int = 30,
        guidance_scale: float = 7.5,
        strength: float = 0.65,
        identity_strength: float = 0.8,
        conditioning_scale: float = 0.8,
        structure_strength: float = 0.35,
        preserve_source_aspect: bool = True,
        width: int | None = None,
        height: int | None = None,
        seed: int | None = None,
    ) -> Any:
        image.file.seek(0)
        filename = image.filename or "input.png"
        content_type = image.content_type or "image/png"
        extension = Path(filename).suffix.lower()
        if not extension:
            filename = f"{filename}.png"

        data = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "model_key": model_key,
            "sampler": sampler,
            "steps": str(steps),
            "guidance_scale": str(guidance_scale),
            "strength": str(strength),
            "identity_strength": str(identity_strength),
            "conditioning_scale": str(conditioning_scale),
            "structure_strength": str(structure_strength),
            "preserve_source_aspect": str(preserve_source_aspect).lower(),
        }
        if width is not None:
            data["width"] = str(width)
        if height is not None:
            data["height"] = str(height)
        if seed is not None:
            data["seed"] = str(seed)

        files = {
            "image": (filename, image.file, content_type),
        }

        return await self._request_json(
            "POST",
            "/generation/edit",
            timeout=120.0,
            data=data,
            files=files,
        )

    async def get_result(self, task_id: str) -> Any:
        return await self._request_json(
            "GET",
            f"/generation/tasks/{task_id}",
            timeout=15.0,
        )

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        timeout: float,
        **kwargs: Any,
    ) -> Any:
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.request(
                    method,
                    f"{self.base_url}{path}",
                    **kwargs,
                )
        except httpx.RequestError as exc:
            raise HTTPException(
                status_code=502,
                detail="Generation service is unavailable.",
            ) from exc

        if response.is_error:
            raise self._build_upstream_error(response)

        try:
            return response.json()
        except ValueError as exc:
            raise HTTPException(
                status_code=502,
                detail="Generation service returned invalid JSON.",
            ) from exc

    def _build_upstream_error(self, response: httpx.Response) -> HTTPException:
        detail: Any = "Generation service request failed."
        try:
            payload = response.json()
        except ValueError:
            payload = None

        if isinstance(payload, dict) and "detail" in payload:
            detail = payload["detail"]
        elif response.text:
            detail = response.text

        return HTTPException(status_code=response.status_code, detail=detail)
