from typing import Any, Dict, List, Optional, TypedDict

from jet.adapters.llama_cpp.config import LLM_BASE_URL
from jet.logger import logger
from openai import OpenAI


def get_llama_cpp_base_url(override: Optional[str] = None) -> str:
    """Return base URL for llama.cpp LLM server (no /v1).

    Args:
        override: Direct URL override (highest priority)
    """
    if override:
        base = override
    else:
        base = LLM_BASE_URL

    if not base:
        base = "http://localhost:8080"

    base = base.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3].rstrip("/")
    return base


class ModelInfo(TypedDict):
    id: str
    object: str
    created: int
    owned_by: str
    meta: Optional[Dict[str, Any]]


class ModelsResponse(TypedDict):
    object: str
    data: List[ModelInfo]


def get_models(base_url: Optional[str] = None) -> ModelsResponse:
    """
    Get loaded model(s) via OpenAI-compatible /v1/models.

    Args:
        base_url: Direct URL override (highest priority)
    """
    url = get_llama_cpp_base_url(override=base_url)
    client = OpenAI(base_url=f"{url}/v1", api_key="not-needed")
    logger.info(f"Fetching models from {url}")
    models = client.models.list()
    logger.debug(f"Retrieved {len(models.data)} model(s)")
    return models.model_dump()


if __name__ == "__main__":
    print(f"{'=' * 60}")
    print("Server: llm")
    print(f"{'=' * 60}")
    try:
        models = get_models()
        model_count = len(models["data"])
        if model_count == 0:
            print("  ⚠️  No models loaded")
        else:
            print(f"  ✅ Found {model_count} model(s):")
            for model in models["data"]:
                print(f"\n  Model: {model['id']}")
                for key, value in model.items():
                    if key != "id":
                        print(f"    {key}: {value}")
    except Exception as e:
        print(f"  ❌ Failed to fetch models: {e}")
