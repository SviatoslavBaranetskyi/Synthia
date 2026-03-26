from functools import lru_cache
from llm_service.llm_client import LLMClient
from moderation_service.router.hybrid_router import HybridRouter
from moderation_service.mod_config import settings


@lru_cache
def get_router():
    llm_client = LLMClient(
        model=settings.LLM_MODEL,
        api_base=settings.LLM_API_BASE,
        api_key=settings.LLM_API_KEY,
    )

    return HybridRouter(
        model_path=settings.MODEL_PATH,
        embedding_index_path=settings.EMBEDDING_INDEX_PATH,
        llm_client=llm_client,
    )