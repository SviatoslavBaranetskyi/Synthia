import httpx

from api_gateway.api_config import settings
from schemas.generation import GenerateRequest


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

    async def get_result(self, task_id: str):
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(f"{self.base_url}/generation/tasks/{task_id}")
            response.raise_for_status()
            return response.json()
