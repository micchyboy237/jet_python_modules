"""Unified configuration for crawl4ai_lib.

Reuses llama_cpp adapter configs to maintain consistency across the codebase.
All environment variable names match the adapter convention (LLAMA_CPP_*).
"""

import os

from jet.adapters.llama_cpp.config import (
    EMBED_BASE_URL,
    EMBED_MODEL,
    LLM_BASE_URL,
    LLM_MODEL,
    RERANK_BASE_URL,
    RERANK_MODEL,
)

# ---------------------------------------------------------------------------
# Crawler-specific defaults
# ---------------------------------------------------------------------------
SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:8888")
DEFAULT_TOP_K = int(os.getenv("CRAWL_TOP_K", "5"))
DEFAULT_MAX_SEARCH_RESULTS = int(os.getenv("CRAWL_MAX_SEARCH_RESULTS", "10"))
DEFAULT_MAX_RETRIES = int(os.getenv("CRAWL_MAX_RETRIES", "3"))
BM25_THRESHOLD = float(os.getenv("CRAWL_BM25_THRESHOLD", "0.6"))
EMBED_REQUEST_TIMEOUT = float(os.getenv("CRAWL_EMBED_TIMEOUT", "30.0"))
SEARXNG_REQUEST_TIMEOUT = float(os.getenv("CRAWL_SEARXNG_TIMEOUT", "12.0"))

# ---------------------------------------------------------------------------
# Re-export adapter configs for single-import convenience
# ---------------------------------------------------------------------------
__all__ = [
    # Adapter configs (re-exported)
    "EMBED_BASE_URL",
    "EMBED_MODEL",
    "LLM_BASE_URL",
    "LLM_MODEL",
    "RERANK_BASE_URL",
    "RERANK_MODEL",
    # Crawler-specific configs
    "SEARXNG_URL",
    "DEFAULT_TOP_K",
    "DEFAULT_MAX_SEARCH_RESULTS",
    "DEFAULT_MAX_RETRIES",
    "BM25_THRESHOLD",
    "EMBED_REQUEST_TIMEOUT",
    "SEARXNG_REQUEST_TIMEOUT",
]
