# jet/adapters/llama_cpp/chunk_strategies/_common.py
"""Shared types, protocols, and helpers for chunking strategies."""

from typing import List, Protocol, Tuple

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


def detect_token_overlap(
    prev_tokens: List[int],
    curr_tokens: List[int],
) -> int:
    """Count overlapping tokens between two consecutive chunks via token ID matching.

    Finds the longest suffix of prev_tokens that matches a prefix of curr_tokens.
    Suitable for fixed-size chunking where overlap is mechanically exact.

    Args:
        prev_tokens: Token IDs of the previous chunk.
        curr_tokens: Token IDs of the current chunk.

    Returns:
        Number of overlapping tokens (0 if none found).
    """
    if not prev_tokens or not curr_tokens:
        return 0

    max_check = min(len(prev_tokens), len(curr_tokens))
    for length in range(max_check, 0, -1):
        if prev_tokens[-length:] == curr_tokens[:length]:
            return length
    return 0


def detect_text_overlap(
    prev_text: str,
    curr_text: str,
    size_fn: TokenSizeFn,
) -> Tuple[str | None, int]:
    """Find overlapping text between consecutive sentence-based chunks.

    Sentence chunker overlap produces exact text duplicates (full sentences
    carried forward). Uses string-level suffix matching since BPE tokenization
    of independently-decoded chunks may produce different token IDs for the
    same text.

    Args:
        prev_text: Text of the previous chunk.
        curr_text: Text of the current chunk.
        size_fn: Tokenizer size function for counting overlap tokens.

    Returns:
        Tuple of (overlap_text, overlap_token_count). Returns (None, 0) if
        no overlap detected.
    """
    if not prev_text.strip() or not curr_text.strip():
        return None, 0

    prev_stripped = prev_text.rstrip()
    curr_stripped = curr_text.lstrip()

    # Full containment: entire curr is a suffix of prev
    if prev_stripped.endswith(curr_stripped):
        tok_count = len(size_fn(curr_stripped))
        return curr_stripped, tok_count

    # Word-boundary suffix matching
    words = curr_stripped.split()
    for start_idx in range(len(words)):
        candidate = " ".join(words[start_idx:])
        if candidate and prev_stripped.endswith(candidate):
            tok_count = len(size_fn(candidate))
            return candidate, tok_count

    # Sentence-level fallback
    from jet.wordnet.sentence import split_sentences

    curr_sents = split_sentences(curr_stripped)
    for sent in curr_sents:
        sent_clean = sent.strip()
        if sent_clean and prev_stripped.endswith(sent_clean):
            tok_count = len(size_fn(sent_clean))
            return sent_clean, tok_count

    return None, 0
