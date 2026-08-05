# jet/adapters/llama_cpp/chunk_strategies/fixed_size_chunker.py
"""Token-aware fixed-size sliding window chunking strategy."""

from typing import List

from jet.adapters.llama_cpp.chunk_strategies._common import (
    TokenSizeFn,
    effective_size,
    merge_small_tail,
    step_size,
)
from jet.adapters.llama_cpp.chunking_utils import (
    _decode_tokens_batch,
    _get_size_fn,
)
from jet.adapters.llama_cpp.types import LLAMACPP_KEYS


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

    def __init__(self, model: str | LLAMACPP_KEYS) -> None:
        self.model = model
        self.size_fn: TokenSizeFn = _get_size_fn(model)

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

        eff = effective_size(chunk_size, buffer)
        step = step_size(chunk_size, chunk_overlap, buffer)

        tokens = self.size_fn(text)
        if not tokens:
            return []

        total_len = len(tokens)
        boundaries: List[tuple[int, int]] = []
        pos = 0
        while pos < total_len:
            end = min(pos + eff, total_len)
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

        return merge_small_tail(
            chunks, min_chunk_size, chunk_size, self.size_fn, self.model
        )
