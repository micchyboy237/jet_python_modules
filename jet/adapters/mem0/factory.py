from typing import Any, Dict

from jet.adapters.llama_cpp.config import (
    EMBED_BASE_URL,
    EMBED_DIMS,
    EMBED_MODEL,
    LLM_BASE_URL,
    LLM_MODEL,
)
from jet.adapters.llama_cpp.model_utils import get_model_ctx_embd_size
from jet.db.postgres.config import (
    DEFAULT_DB,
    DEFAULT_HOST,
    DEFAULT_PASSWORD,
    DEFAULT_PORT,
    DEFAULT_USER,
)

info = get_model_ctx_embd_size(LLM_MODEL)
DEFAULT_MAX_TOKENS = info["ctx"]


def get_memory_config(
    collection_name: str = "memories_v1",
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0.4,
    reset: bool = False,
) -> Dict[str, Any]:
    config: Dict[str, Any] = {
        "llm": {
            "provider": "openai",
            "config": {
                "model": LLM_MODEL,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "openai_base_url": LLM_BASE_URL,
                "api_key": "dummy",
            },
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": EMBED_MODEL,
                "embedding_dims": EMBED_DIMS,  # fallback 768 if model not in dict
                "openai_base_url": EMBED_BASE_URL,
                "api_key": "dummy",
            },
        },
        "vector_store": {
            "provider": "pgvector",
            "config": {
                "collection_name": collection_name,
                "embedding_model_dims": EMBED_DIMS,
                "dbname": DEFAULT_DB,
                "user": DEFAULT_USER,
                "password": DEFAULT_PASSWORD,
                "host": DEFAULT_HOST,
                "port": DEFAULT_PORT,
            },
        },
    }

    return config
