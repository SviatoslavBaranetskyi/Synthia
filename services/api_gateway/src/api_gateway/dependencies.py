from functools import lru_cache

from api_gateway.clients.generation_client import GenerationClient
from api_gateway.clients.moderation_client import ModerationClient


@lru_cache
def get_moderation_client():
    return ModerationClient()


@lru_cache
def get_generation_client():
    return GenerationClient()
