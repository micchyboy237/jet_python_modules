# rag_module_v1/config.py

from dataclasses import dataclass


@dataclass(frozen=True)
class RAGConfig:
    vector_top_k: int = 20
    bm25_top_k: int = 20
    fusion_top_k: int = 20
    rerank_top_n: int = 10

    vector_min_score: float | None = None
    bm25_min_score: float = 0.01

    default_abstention_threshold: float = 0.55
    min_absolute_threshold: float = 0.50
    zero_variance_margin: float = 0.05

    max_context_tokens: int = 2000
    max_query_chars: int = 1000
    max_thought_context_chars: int = 4000

    enable_query_rewrite: bool = True
    enable_metadata_extraction: bool = True
    enable_rerank: bool = True
