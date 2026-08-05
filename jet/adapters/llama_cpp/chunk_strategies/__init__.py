"""Reusable RAG chunking strategies for token-constrained local LLMs.
Public API (stable, re-exported for backward compatibility):
    - ChunkStrategy: Protocol that all strategies implement
    - TokenAwareSentenceChunker: Sentence-first hierarchical chunking
    - TokenAwareFixedSizeChunker: Token-level sliding window chunking
    - SmartChunker: Automatic strategy selection based on document structure
    - ParentDocumentChunker: Parent-document retrieval with linked child-parent pairs
    - get_chunker: Factory function to retrieve a strategy by name
    - detect_token_overlap: Token ID-based overlap detection (fixed-size)
    - detect_text_overlap: String-based overlap detection (sentence)
    - get_optimal_chunk_size: Determine optimal chunk size for a model
    - estimate_tokens_safe: Conservative token estimation for input text
    - format_chunks_for_rag: Format chunks for retrieval-augmented generation
    - detect_content_type: Heuristic content type detection for RAG
"""

from jet.adapters.llama_cpp.chunk_strategies._common import (
    ChunkStrategy,
    detect_text_overlap,
    detect_token_overlap,
)
from jet.adapters.llama_cpp.chunk_strategies.fixed_size_chunker import (
    TokenAwareFixedSizeChunker,
)
from jet.adapters.llama_cpp.chunk_strategies.model_utils import (
    estimate_tokens_safe,
    get_optimal_chunk_size,
)
from jet.adapters.llama_cpp.chunk_strategies.parent_document_chunker import (
    ParentDocumentChunker,
)
from jet.adapters.llama_cpp.chunk_strategies.rag_formatter import (
    detect_content_type,
    format_chunks_for_rag,
)
from jet.adapters.llama_cpp.chunk_strategies.sentence_chunker import (
    TokenAwareSentenceChunker,
)
from jet.adapters.llama_cpp.chunk_strategies.smart_chunker import SmartChunker

_STRATEGY_REGISTRY: dict[str, type[ChunkStrategy]] = {
    "sentence": TokenAwareSentenceChunker,
    "fixed": TokenAwareFixedSizeChunker,
    "smart": SmartChunker,
    "pdr": ParentDocumentChunker,
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
            - "smart": SmartChunker (auto-detects structure)
            - "pdr": ParentDocumentChunker (parent-document retrieval pairs)
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
    "SmartChunker",
    "ParentDocumentChunker",
    "get_chunker",
    "detect_token_overlap",
    "detect_text_overlap",
    "get_optimal_chunk_size",
    "estimate_tokens_safe",
    "format_chunks_for_rag",
    "detect_content_type",
]
