# jet/adapters/llama_cpp/factory.py
"""Centralized client factory for llama.cpp OpenAI-compatible servers.

Provides singleton-style client constructors for LLM, Embedding, Rerank,
and Vision endpoints. All clients use AsyncOpenAI/OpenAI with consistent
timeout, retry, and auth configuration.
"""

from __future__ import annotations

import os

from jet.adapters.llama_cpp.config import (
    EMBED_BASE_URL,
    LLM_BASE_URL,
    RERANK_BASE_URL,
    VISION_BASE_URL,
)
from openai import AsyncOpenAI, OpenAI
from openai.resources import Embeddings

# Shared defaults
_DEFAULT_TIMEOUT = 120.0
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_API_KEY = "not-needed"


def get_llm_client(
    base_url: str = LLM_BASE_URL,
    api_key: str = _DEFAULT_API_KEY,
    timeout: float = _DEFAULT_TIMEOUT,
    max_retries: int = _DEFAULT_MAX_RETRIES,
) -> OpenAI:
    """Synchronous OpenAI client for the LLM server."""
    return OpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        max_retries=max_retries,
    )


def get_async_llm_client(
    base_url: str = LLM_BASE_URL,
    api_key: str = _DEFAULT_API_KEY,
    timeout: float = _DEFAULT_TIMEOUT,
    max_retries: int = _DEFAULT_MAX_RETRIES,
) -> AsyncOpenAI:
    """Async OpenAI client for the LLM server.

    Used by llm_utils.achat() and chat_stream_observability_async.
    """
    return AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        max_retries=max_retries,
    )


def get_embedding_client(
    base_url: str = EMBED_BASE_URL,
    api_key: str = _DEFAULT_API_KEY,
    timeout: float = _DEFAULT_TIMEOUT,
    max_retries: int = _DEFAULT_MAX_RETRIES,
) -> OpenAI:
    """Synchronous OpenAI client for the embedding server."""
    return OpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        max_retries=max_retries,
    )


def get_async_embedding_client(
    base_url: str = EMBED_BASE_URL,
    api_key: str = _DEFAULT_API_KEY,
    timeout: float = _DEFAULT_TIMEOUT,
    max_retries: int = _DEFAULT_MAX_RETRIES,
) -> AsyncOpenAI:
    """Async OpenAI client for the embedding server."""
    return AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        max_retries=max_retries,
    )


def get_embeddings(
    base_url: str = EMBED_BASE_URL,
    api_key: str = _DEFAULT_API_KEY,
    timeout: float = _DEFAULT_TIMEOUT,
    max_retries: int = _DEFAULT_MAX_RETRIES,
) -> Embeddings:
    """Convenience accessor for the synchronous embeddings resource."""
    client = get_embedding_client(
        base_url=base_url, api_key=api_key, timeout=timeout, max_retries=max_retries
    )
    return client.embeddings


def get_rerank_client(
    base_url: str = RERANK_BASE_URL,
    api_key: str = _DEFAULT_API_KEY,
    timeout: float = _DEFAULT_TIMEOUT,
    max_retries: int = _DEFAULT_MAX_RETRIES,
) -> OpenAI:
    """Synchronous OpenAI client for the rerank server."""
    _base_url = base_url or os.getenv(
        "LLAMA_CPP_RERANK_URL", "http://localhost:8082/v1"
    )
    return OpenAI(
        base_url=_base_url,
        api_key=api_key,
        timeout=timeout,
        max_retries=max_retries,
    )


def get_async_rerank_client(
    base_url: str = RERANK_BASE_URL,
    api_key: str = _DEFAULT_API_KEY,
    timeout: float = _DEFAULT_TIMEOUT,
    max_retries: int = _DEFAULT_MAX_RETRIES,
) -> AsyncOpenAI:
    """Async OpenAI client for the rerank server."""
    _base_url = base_url or os.getenv(
        "LLAMA_CPP_RERANK_URL", "http://localhost:8082/v1"
    )
    return AsyncOpenAI(
        base_url=_base_url,
        api_key=api_key,
        timeout=timeout,
        max_retries=max_retries,
    )


def get_vision_client(
    base_url: str = VISION_BASE_URL,
    api_key: str = _DEFAULT_API_KEY,
    timeout: float = _DEFAULT_TIMEOUT,
    max_retries: int = _DEFAULT_MAX_RETRIES,
) -> OpenAI:
    """Synchronous OpenAI client for the vision/multimodal server."""
    _base_url = base_url or LLM_BASE_URL
    return OpenAI(
        base_url=_base_url,
        api_key=api_key,
        timeout=timeout,
        max_retries=max_retries,
    )


def get_async_vision_client(
    base_url: str = VISION_BASE_URL,
    api_key: str = _DEFAULT_API_KEY,
    timeout: float = _DEFAULT_TIMEOUT,
    max_retries: int = _DEFAULT_MAX_RETRIES,
) -> AsyncOpenAI:
    """Async OpenAI client for the vision/multimodal server."""
    _base_url = base_url or LLM_BASE_URL
    return AsyncOpenAI(
        base_url=_base_url,
        api_key=api_key,
        timeout=timeout,
        max_retries=max_retries,
    )
