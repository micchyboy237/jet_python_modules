import os
from typing import Literal

from crawl4ai import AdaptiveConfig, LLMConfig


def get_llm_config(
    *,
    strategy: Literal["statistical", "embedding", "llm"] = "statistical",
    **kwargs,
):
    settings = {**kwargs}
    if strategy == "embedding":
        settings = {
            "provider": "openai/nomic-embed-text-v2-moe",
            "base_url": os.getenv("LLAMA_CPP_EMBED_URL"),
            **settings,
            "max_tokens": settings["max_tokens"]
            if settings.get("max_tokens") is not None
            else 2048,
        }
    elif strategy == "llm":
        settings = {
            "provider": "openai/qwen3-instruct-2507:4b",
            "base_url": os.getenv("LLAMA_CPP_LLM_URL"),
            **settings,
            "temperature": settings["temperature"]
            if settings.get("temperature") is not None
            else 0.7,
            "max_tokens": settings["max_tokens"]
            if settings.get("max_tokens") is not None
            else 12000,
        }

    config = LLMConfig(**settings)
    return config


def get_adaptive_config(
    *,
    strategy: Literal["statistical", "embedding", "llm"] = "statistical",
    temperature: float | None = None,
    max_tokens: int | None = None,
    **kwargs,
) -> AdaptiveConfig:
    config = get_llm_config(
        strategy=strategy, temperature=temperature, max_tokens=max_tokens
    )

    settings = {"embedding_llm_config": config, **kwargs}
    adaptive_config = AdaptiveConfig(**settings)
    return adaptive_config
