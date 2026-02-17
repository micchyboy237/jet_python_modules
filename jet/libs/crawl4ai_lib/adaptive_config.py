import os
from typing import Literal

from crawl4ai import AdaptiveConfig, LLMConfig


def get_llm_config(
    *,
    strategy: Literal["statistical", "embedding", "llm"] = "statistical",
    temperature=0.7,
    **kwargs,
):
    settings = {**kwargs}
    if strategy == "embedding":
        settings = {
            "provider": "openai/nomic-embed-text-v2-moe",
            "base_url": os.getenv("LLAMA_CPP_EMBED_URL"),
            "max_tokens": 2048,
            **settings,
        }
    elif strategy == "llm":
        settings = {
            "provider": "openai/qwen3-instruct-2507:4b",
            "base_url": os.getenv("LLAMA_CPP_LLM_URL"),
            "temperature": temperature,
            "max_tokens": 12000,
            **settings,
        }

    config = LLMConfig(**settings)
    return config


def get_adaptive_config(
    *,
    strategy: Literal["statistical", "embedding", "llm"] = "statistical",
    temperature=0.7,
    **kwargs,
):
    config = get_llm_config(strategy=strategy, temperature=temperature)

    settings = {"embedding_llm_config": config, **kwargs}
    adaptive_config = AdaptiveConfig(**settings)
    return adaptive_config
