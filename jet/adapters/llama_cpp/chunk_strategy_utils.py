# jet_python_modules/jet/adapters/llama_cpp/chunk_strategy_utils.py
"""Reusable RAG chunking strategies for token-constrained local LLMs.

Provides two primary strategies extracted from chunking_utils:
1. TokenAwareSentenceChunker: Hierarchical sentence-first chunking with token-exact sizing
2. TokenAwareFixedSizeChunker: Pure token-level sliding window chunking

Both strategies operate in exact token space using the model's native tokenizer,
making them suitable for small max-context llama.cpp servers.
"""

from typing import List, Protocol

from jet.adapters.llama_cpp.chunking_utils import (
    _decode_tokens_batch,
    _get_last_n_tokens_and_decode,
    _get_size_fn,
)
from jet.adapters.llama_cpp.types import LLAMACPP_KEYS
from jet.wordnet.sentence import split_sentences


class ChunkStrategy(Protocol):
    """Protocol that all chunking strategies must implement."""

    def chunk(
        self,
        text: str,
        chunk_size: int,
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


class _TokenSizeFn(Protocol):
    """Internal protocol for the size function returned by _get_size_fn."""

    def __call__(
        self, text: str | list[str], show_progress: bool = False
    ) -> list[int] | list[list[int]]: ...


def _effective_size(chunk_size: int, buffer: int) -> int:
    """Compute effective chunk size after subtracting buffer."""
    return max(1, chunk_size - buffer)


def _step_size(chunk_size: int, chunk_overlap: int, buffer: int) -> int:
    """Compute step size for sliding window."""
    return max(1, chunk_size - chunk_overlap - buffer)


def _merge_small_tail(
    chunks: List[str],
    min_chunk_size: int,
    chunk_size: int,
    size_fn: _TokenSizeFn,
    model: str | LLAMACPP_KEYS,
) -> List[str]:
    """Merge or discard tail chunk if below min_chunk_size.

    Checks for duplicate content via last-n-tokens comparison before merging.
    """
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


class TokenAwareSentenceChunker:
    """Hierarchical chunking that respects sentence boundaries with token-exact sizing.

    Strategy:
    1. Split text into sentences using NLTK-based splitter.
    2. Accumulate sentences until token budget is exhausted.
    3. If a single sentence exceeds budget, fall back to word-level token splitting.
    4. Apply token-based overlap by walking backward through previous sentences.
    5. Merge undersized tail chunks to prevent fragment waste.

    Best for: General prose, articles, documentation where sentence coherence matters.
    """

    def __init__(self, model: str | LLAMACPP_KEYS):
        self.model = model
        self.size_fn: _TokenSizeFn = _get_size_fn(model)

    def chunk(
        self,
        text: str,
        chunk_size: int = 128,
        chunk_overlap: int = 0,
        min_chunk_size: int = 32,
        buffer: int = 0,
    ) -> List[str]:
        if not text.strip():
            return []

        effective = _effective_size(chunk_size, buffer)
        sentences = split_sentences(text)
        if not sentences:
            return []

        sent_tokens = self.size_fn(sentences)
        sent_sizes = [len(t) for t in sent_tokens]

        chunks: List[str] = []
        current_chunk: List[str] = []
        current_size = 0

        for i, (sentence, s_size) in enumerate(zip(sentences, sent_sizes)):
            # Handle oversized sentence via word-level fallback
            if s_size > effective:
                sub_chunks = self._split_large_sentence(sentence, effective)
                for sub in sub_chunks:
                    sub_size = len(self.size_fn(sub))
                    if current_size + sub_size > effective and current_chunk:
                        chunks.append(" ".join(current_chunk))
                        current_chunk, current_size = self._apply_overlap(
                            current_chunk,
                            sentences,
                            sent_sizes,
                            chunk_overlap,
                        )
                        # If overlap alone fills the budget, flush and reset
                        if current_size >= effective:
                            chunks.append(" ".join(current_chunk))
                            current_chunk, current_size = [], 0
                    if current_size + sub_size > effective and current_chunk:
                        chunks.append(" ".join(current_chunk))
                        current_chunk, current_size = [], 0
                    current_chunk.append(sub)
                    current_size += sub_size
            else:
                if current_size + s_size > effective and current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk, current_size = self._apply_overlap(
                        current_chunk,
                        sentences,
                        sent_sizes,
                        chunk_overlap,
                    )
                    # If overlap alone fills the budget, flush and reset
                    if current_size >= effective:
                        chunks.append(" ".join(current_chunk))
                        current_chunk, current_size = [], 0
                # Re-check after overlap injection before appending
                if current_size + s_size > effective and current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk, current_size = [], 0
                current_chunk.append(sentence)
                current_size += s_size

        # Flush remaining
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return _merge_small_tail(
            chunks, min_chunk_size, chunk_size, self.size_fn, self.model
        )

    def _split_large_sentence(self, sentence: str, max_tokens: int) -> List[str]:
        """Split oversized sentence at word boundaries using token counts."""
        words = sentence.split()
        if not words:
            return []

        word_tokens = self.size_fn(words)
        word_sizes = [len(t) for t in word_tokens]

        result: List[str] = []
        current_words: List[str] = []
        current_size = 0

        for word, w_size in zip(words, word_sizes):
            if current_size + w_size > max_tokens and current_words:
                result.append(" ".join(current_words))
                current_words = [word]
                current_size = w_size
            else:
                current_words.append(word)
                current_size += w_size

        if current_words:
            result.append(" ".join(current_words))
        return result

    def _apply_overlap(
        self,
        current_chunk: List[str],
        all_sentences: List[str],
        sent_sizes: List[int],
        chunk_overlap: int,
    ) -> tuple[List[str], int]:
        """Build overlap buffer from tail of previous chunk using token counts.

        Returns overlap sentences and their total token count.
        Does NOT validate against effective size — caller must handle that.
        """
        if chunk_overlap <= 0 or len(current_chunk) <= 1:
            return [], 0

        overlap_sents: List[str] = []
        overlap_len = 0

        for sent in reversed(current_chunk):
            try:
                idx = all_sentences.index(sent)
                tok_len = sent_sizes[idx]
            except ValueError:
                tok_len = len(self.size_fn(sent))

            overlap_len += tok_len
            overlap_sents.insert(0, sent)
            if overlap_len >= chunk_overlap:
                break

        return overlap_sents, overlap_len


class TokenAwareFixedSizeChunker:
    """Pure token-level sliding window chunking without sentence awareness.

    Strategy:
    1. Tokenize entire text upfront.
    2. Slide a fixed-size window across token IDs with configurable step.
    3. Batch-decode all windows for efficiency.
    4. Filter undersized non-terminal chunks.
    5. Merge/discard undersized tail chunk with duplicate detection.

    Best for: Code, structured data, or when sentence boundaries are irrelevant
    and maximum throughput is needed.
    """

    def __init__(self, model: str | LLAMACPP_KEYS):
        self.model = model
        self.size_fn: _TokenSizeFn = _get_size_fn(model)

    def chunk(
        self,
        text: str,
        chunk_size: int = 128,
        chunk_overlap: int = 0,
        min_chunk_size: int = 32,
        buffer: int = 0,
    ) -> List[str]:
        if not text.strip():
            return []

        effective = _effective_size(chunk_size, buffer)
        step = _step_size(chunk_size, chunk_overlap, buffer)

        tokens = self.size_fn(text)
        if not tokens:
            return []

        total_len = len(tokens)
        boundaries: List[tuple[int, int]] = []
        pos = 0
        while pos < total_len:
            end = min(pos + effective, total_len)
            boundaries.append((pos, end))
            pos += step

        if not boundaries:
            return []

        chunk_token_slices = [tokens[s:e] for s, e in boundaries]
        chunk_texts = _decode_tokens_batch(
            chunk_token_slices, self.model, show_progress=False
        )

        chunks: List[str] = []
        for idx, (chunk_text, chunk_tokens) in enumerate(
            zip(chunk_texts, chunk_token_slices)
        ):
            is_last = boundaries[idx][1] >= total_len
            tok_count = len(chunk_tokens)

            if (
                tok_count < min_chunk_size
                and not is_last
                and chunk_size > min_chunk_size
            ):
                continue
            chunks.append(chunk_text)

        return _merge_small_tail(
            chunks, min_chunk_size, chunk_size, self.size_fn, self.model
        )


def get_chunker(
    strategy: str,
    model: str | LLAMACPP_KEYS,
) -> ChunkStrategy:
    """Factory function to retrieve a chunking strategy by name.

    Args:
        strategy: Strategy identifier. One of:
            - "sentence": TokenAwareSentenceChunker (default, recommended for prose)
            - "fixed": TokenAwareFixedSizeChunker (best for code/structured data)
        model: llama.cpp model key or HF ID for tokenizer resolution.

    Returns:
        ChunkStrategy instance ready for use.

    Raises:
        ValueError: If strategy name is unrecognized.
    """
    strategies = {
        "sentence": TokenAwareSentenceChunker,
        "fixed": TokenAwareFixedSizeChunker,
    }
    cls = strategies.get(strategy.lower())
    if cls is None:
        available = ", ".join(sorted(strategies.keys()))
        raise ValueError(
            f"Unknown chunking strategy '{strategy}'. Available: {available}"
        )
    return cls(model)
