# /Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/adapters/llama_cpp/model_utils.py
import os
from typing import Any, Dict, List, Literal, Optional, TypedDict

from jet.adapters.llama_cpp.config import (
    EMBED_BASE_URL,
    LLM_BASE_URL,
    RERANK_BASE_URL,
)
from jet.logger import logger
from openai import OpenAI

# Environment-aware base URL for OpenAI-compatible endpoints
LLM_V1_URL_ENV = os.getenv("LLAMA_CPP_LLM_URL")

ServerType = Literal["llm", "embed", "rerank"]


def get_llama_cpp_base_url(
    server: ServerType = "llm", override: Optional[str] = None
) -> str:
    """Return base URL for any of the three llama.cpp servers (no /v1).

    Args:
        server: Which server to target: 'llm', 'embed', or 'rerank'
        override: Direct URL override (highest priority)
    """
    if override:
        base = override
    else:
        if server == "llm":
            base = LLM_BASE_URL
        elif server == "embed":
            base = EMBED_BASE_URL
        elif server == "rerank":
            base = RERANK_BASE_URL
        else:
            base = LLM_BASE_URL

    if not base:
        # Default fallbacks
        defaults = {
            "llm": "http://localhost:8080",
            "embed": "http://localhost:8081",
            "rerank": "http://localhost:8082",
        }
        base = defaults.get(server, "http://localhost:8080")

    # Clean URL: remove trailing slash and accidental /v1
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


def get_models(
    base_url: Optional[str] = None,
    server: ServerType = "llm",
) -> ModelsResponse:
    """
    Get loaded model(s) via OpenAI-compatible /v1/models.

    Args:
        base_url: Direct URL override (highest priority)
        server: Which server to target: 'llm', 'embed', or 'rerank'
    """
    url = get_llama_cpp_base_url(server=server, override=base_url)
    client = OpenAI(base_url=f"{url}/v1", api_key="not-needed")

    logger.info(f"Fetching models from server '{server}' at {url}")
    models = client.models.list()
    logger.debug(f"Retrieved {len(models.data)} model(s)")

    return models.model_dump()


if __name__ == "__main__":
    # Demonstrate fetching models from each server type
    servers: List[ServerType] = ["llm", "embed", "rerank"]

    for server_type in servers:
        print(f"\n{'=' * 60}")
        print(f"Server: {server_type}")
        print(f"{'=' * 60}")

        try:
            models = get_models(server=server_type)
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
