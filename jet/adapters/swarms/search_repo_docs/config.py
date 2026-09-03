"""Centralized configuration resolved from jet.adapters.llama_cpp.config + CLI overrides."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

REQUIRED_EXTENSIONS = [
    ".md",
    ".mdx",
    ".py",
    ".ipynb",
    ".txt",
    ".rst",
    ".yaml",
    ".yml",
    ".json",
]


@dataclass
class SearchConfig:
    """Immutable configuration for a search run."""

    # Directories & query
    data_dirs: List[str] = field(default_factory=list)
    query: str = ""

    # Model overrides (None = use env config)
    llm_model: Optional[str] = None
    embed_model: Optional[str] = None

    # Retrieval
    top_k: int = 15
    rerank_top_n: int = 5
    use_reranker: bool = True

    # Generation
    use_stream: bool = True
    enable_thinking: bool = False

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200

    @classmethod
    def from_args(cls, args) -> "SearchConfig":
        """Build from parsed argparse Namespace."""
        return cls(
            data_dirs=args.data_dirs,
            query=args.query,
            llm_model=getattr(args, "llm_model", None),
            embed_model=getattr(args, "embed_model", None),
            top_k=args.top_k,
            rerank_top_n=args.rerank_top_n,
            use_reranker=not args.no_reranker,
            use_stream=not args.no_stream,
            enable_thinking=args.enable_thinking,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )

    def resolve_llm(self) -> tuple[str, str]:
        """Return (model_name, api_base) resolved against env config."""
        from jet.adapters.llama_cpp.config import LLM_BASE_URL, LLM_MODEL

        return self.llm_model or LLM_MODEL, LLM_BASE_URL

    def resolve_embed(self) -> tuple[str, str, int, str, str]:
        """Return (model, base_url, dims, query_prefix, doc_prefix)."""
        from jet.adapters.llama_cpp.config import (
            EMBED_BASE_URL,
            EMBED_DIMS,
            EMBED_DOC_PREFIX,
            EMBED_MODEL,
            EMBED_QUERY_PREFIX,
        )

        return (
            self.embed_model or EMBED_MODEL,
            EMBED_BASE_URL,
            EMBED_DIMS,
            EMBED_QUERY_PREFIX,
            EMBED_DOC_PREFIX,
        )

    def resolve_rerank(self) -> tuple[Optional[str], Optional[str]]:
        """Return (base_url, model) or (None, None) if unavailable."""
        from jet.adapters.llama_cpp.config import RERANK_BASE_URL, RERANK_MODEL

        if self.use_reranker and RERANK_BASE_URL and RERANK_MODEL:
            return RERANK_BASE_URL, RERANK_MODEL
        return None, None
