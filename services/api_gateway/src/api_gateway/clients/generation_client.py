from pathlib import Path

import httpx
from fastapi import UploadFile

from api_gateway.api_config import settings
from schemas.generation import GenerateRequest, GenerationModelKey, GenerationSampler


class GenerationClient:
    def __init__(self):
        self.base_url = settings.GENERATION_SERVICE_URL

    async def generate(self, request: GenerateRequest):
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.base_url}/generation/",
                json=request.model_dump(),
            )
            response.raise_for_status()
            return response.json()

    async def list_models(self):
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(f"{self.base_url}/generation/models")
            response.raise_for_status()
            return response.json()

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
    ):
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

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.base_url}/generation/edit",
                data=data,
                files=files,
            )
            response.raise_for_status()
            return response.json()

    async def get_result(self, task_id: str):
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(f"{self.base_url}/generation/tasks/{task_id}")
            response.raise_for_status()
            return response.json()
