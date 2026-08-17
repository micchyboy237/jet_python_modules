# jet_python_modules/jet/adapters/llama_cpp/chunking_utils/chunking.py
import uuid
from typing import Callable

from jet.adapters.llama_cpp.config import LLM_MODEL
from jet.adapters.llama_cpp.types import LLAMACPP_KEYS
from jet.logger import logger
from jet.wordnet.sentence import split_sentences
from tqdm import tqdm

from .tokenization import (
    _decode_tokens_batch,
    _get_last_n_tokens_and_decode,
    _get_size_fn,
)
from .types import ChunkResult


def split_large_sentence(sentence: str, max_size: int, size_fn: Callable) -> list[str]:
    """Split a large sentence into smaller chunks based on token size.

    Uses batch tokenization of all words for efficiency.
    """
    words = sentence.split()
    if not words:
        return []

    word_tokens = size_fn(words)
    word_sizes = [len(t) for t in word_tokens]

    chunks: list[str] = []
    current_words: list[str] = []
    current_size = 0

    for word, word_size in zip(words, word_sizes):
        if current_size + word_size > max_size and current_words:
            chunks.append(" ".join(current_words))
            current_words = [word]
            current_size = word_size
        else:
            current_words.append(word)
            current_size += word_size

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


def chunk_texts(
    texts: str | list[str],
    chunk_size: int = 128,
    chunk_overlap: int = 0,
    model: str | LLAMACPP_KEYS = LLM_MODEL,
    buffer: int = 0,
    strict_sentences: bool = True,
    min_chunk_size: int = 32,
    show_progress: bool = True,
) -> list[str]:
    """Chunk texts into smaller pieces based on token size.

    Optimizations:
    - All tokenization via local HuggingFace tokenizer (batch where possible)
    - Batch tokenization of sentences instead of sequential
    - Batch tokenization of words in split_large_sentence
    - Direct token counts from token IDs (no re-tokenization)

    Args:
        texts: Single text or list of texts to chunk.
        chunk_size: Maximum tokens per chunk.
        chunk_overlap: Number of overlapping tokens between chunks.
        model: Model key for tokenizer (default: LLM_MODEL).
        buffer: Extra space reserved to avoid exceeding chunk_size.
        strict_sentences: If True, respect sentence boundaries.
        min_chunk_size: Minimum tokens for a chunk to be kept.
        show_progress: Show progress bar during chunking.

    Returns:
        List of chunk strings.
    """
    if min_chunk_size > chunk_size:
        min_chunk_size = chunk_size

    if isinstance(texts, str):
        texts = [texts]

    chunked_texts: list[str] = []
    size_fn = _get_size_fn(model)
    effective_chunk_size = chunk_size - buffer
    step = max(1, chunk_size - chunk_overlap - buffer)

    text_iter = tqdm(
        texts, desc="Chunking texts", unit="text", disable=not show_progress
    )

    for text in text_iter:
        sentences = split_sentences(text)
        if not sentences:
            continue

        # Non-strict mode: pure token-based chunking
        if not strict_sentences:
            tokens = size_fn(text)
            total_len = len(tokens)
            if not tokens:
                continue

            chunk_boundaries: list[tuple[int, int]] = []
            j = 0
            while j < total_len:
                end = min(j + effective_chunk_size, total_len)
                chunk_boundaries.append((j, end))
                j += step

            if not chunk_boundaries:
                continue

            all_chunk_tokens = [tokens[s:e] for s, e in chunk_boundaries]
            all_chunk_texts = _decode_tokens_batch(
                all_chunk_tokens, model, show_progress=False
            )

            for idx, (chunk_text, chunk_tokens_list) in enumerate(
                zip(all_chunk_texts, all_chunk_tokens)
            ):
                chunk_tok_count = len(chunk_tokens_list)
                _, end_pos = chunk_boundaries[idx]
                is_last_chunk = end_pos >= total_len

                if (
                    chunk_tok_count < min_chunk_size
                    and not is_last_chunk
                    and chunk_size > min_chunk_size
                ):
                    continue

                chunked_texts.append(chunk_text)

            # Merge small trailing chunk
            if (
                len(chunked_texts) > 1
                and len(size_fn(chunked_texts[-1])) < min_chunk_size
                and chunk_size > min_chunk_size
            ):
                last_chunk = chunked_texts.pop()
                prev_chunk = chunked_texts[-1]
                prev_chunk_last_n = _get_last_n_tokens_and_decode(
                    prev_chunk, len(size_fn(last_chunk)), model
                )
                if last_chunk != prev_chunk_last_n:
                    chunked_texts[-1] = prev_chunk + " " + last_chunk

            continue

        # Strict mode: sentence-aware chunking
        sent_tokens = size_fn(sentences)
        sent_sizes = [len(t) for t in sent_tokens]
        i, n = 0, len(sentences)
        current_chunk: list[str] = []
        current_size = 0

        while i < n:
            s = sentences[i]
            s_size = sent_sizes[i]

            if s_size > effective_chunk_size:
                sub_sentences = split_large_sentence(s, effective_chunk_size, size_fn)
                for sub in sub_sentences:
                    sub_size = len(size_fn(sub))
                    if current_size + sub_size > effective_chunk_size:
                        if current_chunk:
                            chunked_texts.append(" ".join(current_chunk))
                        current_chunk, current_size = [], 0
                    current_chunk.append(sub)
                    current_size += sub_size
            else:
                if current_size + s_size > effective_chunk_size:
                    chunked_texts.append(" ".join(current_chunk))

                    # Handle overlap
                    if chunk_overlap > 0 and len(current_chunk) > 1:
                        overlap_sents: list[str] = []
                        overlap_len = 0
                        for sent in reversed(current_chunk):
                            sent_idx = (
                                sentences.index(sent) if sent in sentences else -1
                            )
                            sent_token_len = (
                                sent_sizes[sent_idx]
                                if sent_idx >= 0
                                else len(size_fn(sent))
                            )
                            overlap_len += sent_token_len
                            overlap_sents.insert(0, sent)
                            if overlap_len >= chunk_overlap:
                                break

                        current_chunk = overlap_sents
                        current_size = (
                            sum(len(t) for t in size_fn(current_chunk))
                            if current_chunk
                            else 0
                        )
                    else:
                        current_chunk, current_size = [], 0

                current_chunk.append(s)
                current_size += s_size

            i += 1

        # Finalize remaining chunk
        if current_chunk:
            last_chunk = " ".join(current_chunk)
            if len(size_fn(last_chunk)) >= min_chunk_size or not chunked_texts:
                chunked_texts.append(last_chunk)
            else:
                chunked_texts[-1] = chunked_texts[-1] + " " + last_chunk

    return chunked_texts


def chunk_texts_with_data(
    texts: str | list[str],
    chunk_size: int = 128,
    chunk_overlap: int = 0,
    model: str | LLAMACPP_KEYS = LLM_MODEL,
    ids: list[str] | None = None,
    buffer: int = 0,
    strict_sentences: bool = True,
    min_chunk_size: int = 32,
    show_progress: bool = True,
) -> list[ChunkResult]:
    """Chunk texts and return rich ChunkResult objects with metadata.

    Optimizations:
    - All tokenization via local HuggingFace tokenizer (batch)
    - Batch tokenize all texts upfront
    - Batch decode all chunks per document
    - Direct token count from token IDs (no re-tokenization)
    - Batch tokenize all sentences per document
    - Batch tokenize words in split_large_sentence

    Args:
        texts: Single text or list of texts to chunk.
        chunk_size: Maximum tokens per chunk.
        chunk_overlap: Number of overlapping tokens between chunks.
        model: Model key for tokenizer (default: LLM_MODEL).
        ids: Optional list of document IDs.
        buffer: Extra space reserved to avoid exceeding chunk_size.
        strict_sentences: If True, respect sentence boundaries.
        min_chunk_size: Minimum tokens for a chunk to be kept.
        show_progress: Show progress bar during chunking.

    Returns:
        List of ChunkResult dicts with metadata.
    """
    if min_chunk_size > chunk_size:
        min_chunk_size = chunk_size

    if isinstance(texts, str):
        texts = [texts]
        doc_indices = [0]
    else:
        doc_indices = list(range(len(texts)))

    chunks: list[ChunkResult] = []
    effective_chunk_size = chunk_size - buffer
    size_fn = _get_size_fn(model)

    batch_tokens = size_fn(texts, show_progress=show_progress)
    token_counts = [len(tokens) for tokens in batch_tokens]
    step = max(1, chunk_size - chunk_overlap - buffer)

    logger.debug(
        f"chunk_texts_with_data vars: effective={effective_chunk_size}, "
        f"min_chunk={min_chunk_size}, step={step}, "
        f"chunk_size={chunk_size}, overlap={chunk_overlap}, buffer={buffer}, "
        f"docs={len(texts)}, "
        f"avg_tokens={sum(token_counts) / max(len(token_counts), 1):.0f}"
    )

    doc_iter = tqdm(
        zip(doc_indices, texts),
        total=len(texts),
        desc="Chunking texts",
        disable=not show_progress,
    )

    for i, (doc_index, text) in enumerate(doc_iter):
        if not text.strip():
            continue

        doc_id = ids[i] if ids and i < len(ids) else str(uuid.uuid4())
        chunk_index = 0

        # Non-strict mode
        if not strict_sentences:
            tokens = batch_tokens[i]
            if not tokens:
                continue

            total_len = len(tokens)
            chunk_boundaries: list[tuple[int, int]] = []
            j = 0
            while j < total_len:
                end = min(j + effective_chunk_size, total_len)
                chunk_boundaries.append((j, end))
                j += step

            if not chunk_boundaries:
                continue

            all_chunk_tokens = [tokens[s:e] for s, e in chunk_boundaries]
            all_chunk_texts = _decode_tokens_batch(
                all_chunk_tokens, model, show_progress=False
            )

            for idx, (chunk_text, (start, end), chunk_tokens_list) in enumerate(
                zip(all_chunk_texts, chunk_boundaries, all_chunk_tokens)
            ):
                chunk_tok_count = len(chunk_tokens_list)
                is_last_chunk = end >= total_len

                if (
                    chunk_tok_count < min_chunk_size
                    and not is_last_chunk
                    and chunk_size > min_chunk_size
                ):
                    continue

                overlap_start_idx = overlap_end_idx = None
                if chunk_overlap > 0 and end < total_len:
                    overlap_start = max(start, end - chunk_overlap)
                    if overlap_start < end:
                        overlap_start_idx = overlap_start
                        overlap_end_idx = end

                chunks.append(
                    {
                        "id": str(uuid.uuid4()),
                        "doc_id": doc_id,
                        "doc_index": doc_index,
                        "chunk_index": chunk_index,
                        "num_tokens": chunk_tok_count,
                        "content": chunk_text,
                        "start_idx": start,
                        "end_idx": end,
                        "line_idx": 0,
                        "overlap_start_idx": overlap_start_idx,
                        "overlap_end_idx": overlap_end_idx,
                    }
                )
                chunk_index += 1

            # Merge small trailing chunk
            if (
                len(chunks) > 1
                and chunks[-1]["num_tokens"] < min_chunk_size
                and chunk_size > min_chunk_size
            ):
                last = chunks.pop()
                prev = chunks[-1]
                prev_last_n = _get_last_n_tokens_and_decode(
                    prev["content"], last["num_tokens"], model
                )
                if last["content"] != prev_last_n.strip():
                    prev["content"] += " " + last["content"]
                    prev["num_tokens"] = len(size_fn(prev["content"]))
                    prev["end_idx"] = last["end_idx"]

            continue

        # Strict mode
        sentences = split_sentences(text)
        if not sentences:
            continue

        sent_tokens = size_fn(sentences)
        sent_sizes = [len(t) for t in sent_tokens]
        current_chunk: list[str] = []
        current_size = 0

        for s, s_size in zip(sentences, sent_sizes):
            if s_size > effective_chunk_size:
                sub_sents = split_large_sentence(s, effective_chunk_size, size_fn)
                for sub in sub_sents:
                    sub_size = len(size_fn(sub))
                    if current_size + sub_size > effective_chunk_size and current_chunk:
                        chunk_content = " ".join(current_chunk)
                        num_tokens = len(size_fn(chunk_content))

                        if num_tokens >= min_chunk_size or not chunks:
                            chunks.append(
                                {
                                    "id": str(uuid.uuid4()),
                                    "doc_id": doc_id,
                                    "doc_index": doc_index,
                                    "chunk_index": chunk_index,
                                    "num_tokens": num_tokens,
                                    "content": chunk_content,
                                    "start_idx": 0,
                                    "end_idx": len(chunk_content),
                                    "line_idx": 0,
                                    "overlap_start_idx": None,
                                    "overlap_end_idx": None,
                                }
                            )
                            chunk_index += 1

                        current_chunk, current_size = [], 0

                    current_chunk.append(sub)
                    current_size += sub_size
            else:
                if current_size + s_size > effective_chunk_size and current_chunk:
                    chunk_content = " ".join(current_chunk)
                    num_tokens = len(size_fn(chunk_content))

                    if num_tokens >= min_chunk_size or not chunks:
                        chunks.append(
                            {
                                "id": str(uuid.uuid4()),
                                "doc_id": doc_id,
                                "doc_index": doc_index,
                                "chunk_index": chunk_index,
                                "num_tokens": num_tokens,
                                "content": chunk_content,
                                "start_idx": 0,
                                "end_idx": len(chunk_content),
                                "line_idx": 0,
                                "overlap_start_idx": None,
                                "overlap_end_idx": None,
                            }
                        )
                        chunk_index += 1

                    current_chunk, current_size = [], 0

                current_chunk.append(s)
                current_size += s_size

        # Finalize remaining chunk
        if current_chunk:
            chunk_content = " ".join(current_chunk)
            num_tokens = len(size_fn(chunk_content))

            if num_tokens < min_chunk_size and chunks:
                chunks[-1]["content"] += " " + chunk_content
                chunks[-1]["num_tokens"] = len(size_fn(chunks[-1]["content"]))
                chunks[-1]["end_idx"] = len(chunks[-1]["content"])
            else:
                chunks.append(
                    {
                        "id": str(uuid.uuid4()),
                        "doc_id": doc_id,
                        "doc_index": doc_index,
                        "chunk_index": chunk_index,
                        "num_tokens": num_tokens,
                        "content": chunk_content,
                        "start_idx": 0,
                        "end_idx": len(chunk_content),
                        "line_idx": 0,
                        "overlap_start_idx": None,
                        "overlap_end_idx": None,
                    }
                )

    return chunks
