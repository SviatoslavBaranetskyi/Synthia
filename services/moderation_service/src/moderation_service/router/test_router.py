import os
import logging
import warnings

# === ПОДАВЛЕНИЕ ВАРНИНГОВ ===
# 1. Отключаем логи transformers
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

# 2. Отключаем specific warnings
warnings.filterwarnings("ignore", message=".*unexpected.*")
warnings.filterwarnings("ignore", message=".*unauthenticated.*")

# 3. Устанавливаем HF_TOKEN (если есть)
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")  # или вставьте токен

# 4. Опционально - отключаем прогресс-бары
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from moderation_service.router.hybrid_router import HybridRouter
from llm_service.llm_client import LLMClient


def main():
    router = HybridRouter(
        model_path="models/final",
        embedding_index_path="models/embedding_index",
        llm_client=LLMClient(),
    )

    # message = "you are not very smart"
    # message = "wow that was genius... not"
    # message = "great job ruining everything"
    message = "you are kinda weird"

    print("Message:", message)

    result = router.route(message)

    print(result)


if __name__ == "__main__":
    main()