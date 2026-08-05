# jet/adapters/llama_cpp/chunk_strategies/__init__.py
"""Reusable RAG chunking strategies for token-constrained local LLMs.

Public API (stable, re-exported for backward compatibility):
    - ChunkStrategy: Protocol that all strategies implement
    - TokenAwareSentenceChunker: Sentence-first hierarchical chunking
    - TokenAwareFixedSizeChunker: Token-level sliding window chunking
    - get_chunker: Factory function to retrieve a strategy by name
    - detect_token_overlap: Token ID-based overlap detection (fixed-size)
    - detect_text_overlap: String-based overlap detection (sentence)
"""

from jet.adapters.llama_cpp.chunk_strategies._common import (
    ChunkStrategy,
    detect_text_overlap,
    detect_token_overlap,
)
from jet.adapters.llama_cpp.chunk_strategies.fixed_size_chunker import (
    TokenAwareFixedSizeChunker,
)
from jet.adapters.llama_cpp.chunk_strategies.sentence_chunker import (
    TokenAwareSentenceChunker,
)

_STRATEGY_REGISTRY: dict[str, type[ChunkStrategy]] = {
    "sentence": TokenAwareSentenceChunker,
    "fixed": TokenAwareFixedSizeChunker,
}


def get_chunker(
    strategy: str,
    model: str,
) -> ChunkStrategy:
    """Factory function to retrieve a chunking strategy by name.

    Args:
        strategy: Strategy identifier. One of:
            - "sentence": TokenAwareSentenceChunker (recommended for prose)
            - "fixed": TokenAwareFixedSizeChunker (best for code/structured data)
        model: llama.cpp model key or HF ID for tokenizer resolution.

    Returns:
        ChunkStrategy instance ready for use.

    Raises:
        ValueError: If strategy name is unrecognized.
    """
    cls = _STRATEGY_REGISTRY.get(strategy.lower())
    if cls is None:
        available = ", ".join(sorted(_STRATEGY_REGISTRY.keys()))
        raise ValueError(
            f"Unknown chunking strategy '{strategy}'. Available: {available}"
        )
    return cls(model)


__all__ = [
    "ChunkStrategy",
    "TokenAwareSentenceChunker",
    "TokenAwareFixedSizeChunker",
    "get_chunker",
    "detect_token_overlap",
    "detect_text_overlap",
]
