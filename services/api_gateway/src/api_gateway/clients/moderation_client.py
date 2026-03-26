import httpx
from api_gateway.api_config import settings


class ModerationClient:
    def __init__(self):
        self.base_url = settings.MODERATION_SERVICE_URL

    async def moderate(self, text: str):
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                f"{self.base_url}/moderate",
                json={"text": text},
            )
            response.raise_for_status()
            return response.json()

    async def moderate_batch(self, texts: list[str]):
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{self.base_url}/moderate/batch",
                json={"texts": texts},
            )
            response.raise_for_status()
            return response.json()