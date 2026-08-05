# jet/adapters/llama_cpp/chunk_strategy_utils/sentence_chunker.py
"""Token-aware sentence-first hierarchical chunking strategy."""

from typing import List

from jet.adapters.llama_cpp.chunk_strategy_utils._common import (
    TokenSizeFn,
    effective_size,
    merge_small_tail,
)
from jet.adapters.llama_cpp.chunking_utils import _get_size_fn
from jet.adapters.llama_cpp.types import LLAMACPP_KEYS
from jet.wordnet.sentence import split_sentences


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
        sentences = split_sentences(text)
        if not sentences:
            return []

        sent_tokens = self.size_fn(sentences)
        sent_sizes = [len(t) for t in sent_tokens]

        chunks: List[str] = []
        current_chunk: List[str] = []
        current_size = 0

        for sentence, s_size in zip(sentences, sent_sizes):
            # Handle oversized sentence via word-level fallback
            if s_size > eff:
                sub_chunks = self._split_large_sentence(sentence, eff)
                for sub in sub_chunks:
                    sub_size = len(self.size_fn(sub))
                    if current_size + sub_size > eff and current_chunk:
                        chunks.append(" ".join(current_chunk))
                        current_chunk, current_size = self._apply_overlap(
                            current_chunk,
                            sentences,
                            sent_sizes,
                            chunk_overlap,
                        )
                        # If overlap alone fills the budget, flush and reset
                        if current_size >= eff:
                            chunks.append(" ".join(current_chunk))
                            current_chunk, current_size = [], 0
                    if current_size + sub_size > eff and current_chunk:
                        chunks.append(" ".join(current_chunk))
                        current_chunk, current_size = [], 0
                    current_chunk.append(sub)
                    current_size += sub_size
            else:
                if current_size + s_size > eff and current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk, current_size = self._apply_overlap(
                        current_chunk,
                        sentences,
                        sent_sizes,
                        chunk_overlap,
                    )
                    # If overlap alone fills the budget, flush and reset
                    if current_size >= eff:
                        chunks.append(" ".join(current_chunk))
                        current_chunk, current_size = [], 0
                # Re-check after overlap injection before appending
                if current_size + s_size > eff and current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk, current_size = [], 0
                current_chunk.append(sentence)
                current_size += s_size

        # Flush remaining
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return merge_small_tail(
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
