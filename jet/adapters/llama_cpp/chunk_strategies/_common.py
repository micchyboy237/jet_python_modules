# jet/adapters/llama_cpp/chunk_strategies/_common.py
"""Shared types, protocols, and helpers for chunking strategies."""

from typing import List, Protocol

from jet.adapters.llama_cpp.types import LLAMACPP_KEYS


class ChunkStrategy(Protocol):
    """Protocol that all chunking strategies must implement."""

    def chunk(
        self,
        text: str,
        chunk_size: int = 128,
        chunk_overlap: int = 0,
        min_chunk_size: int = 32,
        buffer: int = 0,
    ) -> List[str]:
        """Split text into chunks respecting the strategy's rules.

        Args:
            text: Input text to chunk.
            chunk_size: Maximum tokens per chunk.
            chunk_overlap: Number of overlapping tokens between consecutive chunks.
            min_chunk_size: Minimum tokens for a chunk to be kept (merged otherwise).
            buffer: Extra token margin reserved to avoid exceeding chunk_size.

        Returns:
            List of chunk strings.
        """
        ...


class TokenSizeFn(Protocol):
    """Internal protocol for the size function returned by _get_size_fn."""

    def __call__(
        self,
        text: str | list[str],
        show_progress: bool = False,
    ) -> list[int] | list[list[int]]: ...


def effective_size(chunk_size: int, buffer: int) -> int:
    """Compute effective chunk size after subtracting buffer."""
    return max(1, chunk_size - buffer)


def step_size(chunk_size: int, chunk_overlap: int, buffer: int) -> int:
    """Compute step size for sliding window."""
    return max(1, chunk_size - chunk_overlap - buffer)


def merge_small_tail(
    chunks: List[str],
    min_chunk_size: int,
    chunk_size: int,
    size_fn: TokenSizeFn,
    model: str | LLAMACPP_KEYS,
) -> List[str]:
    """Merge or discard tail chunk if below min_chunk_size.

    Checks for duplicate content via last-n-tokens comparison before merging.
    Shared by both strategies to ensure consistent tail behavior.
    """
    from jet.adapters.llama_cpp.chunking_utils import _get_last_n_tokens_and_decode

    if len(chunks) <= 1 or chunk_size <= min_chunk_size:
        return chunks

    last_tokens = size_fn(chunks[-1])
    if len(last_tokens) >= min_chunk_size:
        return chunks

    last_text = chunks.pop()
    prev_last_n = _get_last_n_tokens_and_decode(chunks[-1], len(last_tokens), model)
    if last_text != prev_last_n.strip():
        chunks[-1] = chunks[-1] + " " + last_text
    return chunks
