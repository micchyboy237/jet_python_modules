import uuid
from typing import Callable, TypedDict, overload

from jet.adapters.llama_cpp.config import LLM_MODEL
from jet.adapters.llama_cpp.token_utils import (
    get_tokenizer,
)
from jet.adapters.llama_cpp.types import LLAMACPP_KEYS
from jet.logger import logger
from jet.wordnet.sentence import split_sentences
from tqdm import tqdm

LOCAL_BATCH_SIZE = 64  # Optimal batch size for local tokenizer operations


# ---------------------------------------------------------------------------
# Internal helpers — all tokenization goes through token_utils.get_tokenizer
# ---------------------------------------------------------------------------
def _tokenize_for_size(text: str, model: str | LLAMACPP_KEYS = LLM_MODEL) -> list[int]:
    """Tokenize single text and return token IDs for size counting."""
    tokenizer = get_tokenizer(model)
    return tokenizer.encode(text, add_special_tokens=False)


def _tokenize_batch_for_size(
    texts: list[str],
    model: str | LLAMACPP_KEYS = LLM_MODEL,
    show_progress: bool = False,
) -> list[list[int]]:
    """Tokenize multiple texts using the tokenizer's __call__ for efficiency."""
    if not texts:
        return []

    tokenizer = get_tokenizer(model)
    results: list[list[int]] = []

    text_iter = tqdm(
        range(0, len(texts), LOCAL_BATCH_SIZE),
        desc="Batch tokenizing",
        unit="batch",
        disable=not show_progress,
    )
    for i in text_iter:
        batch = texts[i : i + LOCAL_BATCH_SIZE]
        # Use __call__ instead of batch_encode_plus (compatible with all backends)
        encoded = tokenizer(batch, add_special_tokens=False)
        # .input_ids works on both BatchEncoding and dict-like returns
        if hasattr(encoded, "input_ids"):
            results.extend(encoded.input_ids)
        else:
            results.extend(encoded["input_ids"])

    return results


def _decode_tokens(tokens: list[int], model: str | LLAMACPP_KEYS = LLM_MODEL) -> str:
    """Decode token IDs back to text."""
    tokenizer = get_tokenizer(model)
    return tokenizer.decode(
        tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )


def _decode_tokens_batch(
    token_lists: list[list[int]],
    model: str | LLAMACPP_KEYS = LLM_MODEL,
    show_progress: bool = False,
) -> list[str]:
    """Batch decode multiple token lists to text."""
    if not token_lists:
        return []

    tokenizer = get_tokenizer(model)
    results: list[str] = []

    text_iter = tqdm(
        range(0, len(token_lists), LOCAL_BATCH_SIZE),
        desc="Batch decoding",
        unit="batch",
        disable=not show_progress,
    )
    for i in text_iter:
        batch = token_lists[i : i + LOCAL_BATCH_SIZE]
        batch_results = tokenizer.batch_decode(
            batch,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        results.extend(batch_results)

    return results


def _get_last_n_tokens_and_decode(
    text: str, n: int, model: str | LLAMACPP_KEYS = LLM_MODEL
) -> str:
    """Get the last n tokens from text and decode them back to string."""
    if n <= 0:
        return ""

    tokenizer = get_tokenizer(model)
    tokens = tokenizer.encode(text, add_special_tokens=False)
    last_n = tokens[-n:] if len(tokens) >= n else tokens
    return tokenizer.decode(
        last_n,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )


def _get_size_fn(model: str | LLAMACPP_KEYS = LLM_MODEL) -> Callable:
    """Return a callable size_fn for chunking.

    Handles both single strings (returns list[int]) and
    lists of strings (returns list[list[int]]).
    Uses tokenizer.__call__() for batch compatibility across all backends.
    """
    tokenizer = get_tokenizer(model)

    def _fn(text, show_progress=False):
        if isinstance(text, list):
            if not text:
                return []
            results = []
            text_iter = tqdm(
                range(0, len(text), LOCAL_BATCH_SIZE),
                desc="Batch tokenizing (size_fn)",
                unit="batch",
                disable=not show_progress,
            )
            for i in text_iter:
                batch = text[i : i + LOCAL_BATCH_SIZE]
                # Use __call__ (compatible with TokenizersBackend, slow, fast tokenizers)
                encoded = tokenizer(batch, add_special_tokens=False)
                if hasattr(encoded, "input_ids"):
                    results.extend(encoded.input_ids)
                else:
                    results.extend(encoded["input_ids"])
            return results
        else:
            return tokenizer.encode(text, add_special_tokens=False)

    return _fn


# ---------------------------------------------------------------------------
# Chunk result type
# ---------------------------------------------------------------------------
class ChunkResult(TypedDict):
    """Core information for an individual chunk.

    Attributes:
        id: Unique chunk identifier.
        doc_id: Document ID (same as in meta).
        doc_index: Document index (same as in meta).
        chunk_index: Chunk order within the document.
        num_tokens: Number of tokens in this chunk.
        content: Text content of the chunk.
        start_idx: Start offset in source content.
        end_idx: End offset in source content.
        line_idx: Line number in the source.
        overlap_start_idx: Start index of overlap with previous chunk, if any.
        overlap_end_idx: End index of overlap with next chunk, if any.
    """

    id: str
    doc_id: str
    doc_index: int
    chunk_index: int
    num_tokens: int
    content: str
    start_idx: int
    end_idx: int
    line_idx: int
    overlap_start_idx: int | None
    overlap_end_idx: int | None


# ---------------------------------------------------------------------------
# Sentence splitting helper
# ---------------------------------------------------------------------------
def split_large_sentence(sentence: str, max_size: int, size_fn: Callable) -> list[str]:
    """Split a large sentence into smaller chunks based on token size.

    Uses batch tokenization of all words for efficiency.
    """
    words = sentence.split()
    if not words:
        return []

    # Batch tokenize all words at once
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


# ---------------------------------------------------------------------------
# Main chunking functions
# ---------------------------------------------------------------------------
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

        # Non-strict path: token-based chunking with batch decode
        if not strict_sentences:
            tokens = size_fn(text)
            total_len = len(tokens)
            if not tokens:
                continue

            # Build all chunk boundaries first
            chunk_boundaries: list[tuple[int, int]] = []
            j = 0
            while j < total_len:
                end = min(j + effective_chunk_size, total_len)
                chunk_boundaries.append((j, end))
                j += step

            if not chunk_boundaries:
                continue

            # Batch decode all chunks at once
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

            # Handle small last chunk merging
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

        # Strict sentence path: respect sentence boundaries
        # Batch tokenize all sentences at once
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

    # Batch tokenize all texts upfront
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

        # Non-strict path: token-based chunking
        if not strict_sentences:
            tokens = batch_tokens[i]
            if not tokens:
                continue

            total_len = len(tokens)

            # Build all chunk boundaries first
            chunk_boundaries: list[tuple[int, int]] = []
            j = 0
            while j < total_len:
                end = min(j + effective_chunk_size, total_len)
                chunk_boundaries.append((j, end))
                j += step

            if not chunk_boundaries:
                continue

            # Batch decode all chunks at once
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

            # Handle small last chunk merging
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

        # Strict sentence path: respect sentence boundaries
        sentences = split_sentences(text)
        if not sentences:
            continue

        # Batch tokenize all sentences
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


@overload
def truncate_texts(
    texts: str,
    model: str | LLAMACPP_KEYS = ...,
    max_tokens: int | None = ...,
    strict_sentences: bool = ...,
    show_progress: bool = ...,
) -> str: ...


@overload
def truncate_texts(
    texts: list[str],
    model: str | LLAMACPP_KEYS = ...,
    max_tokens: int | None = ...,
    strict_sentences: bool = ...,
    show_progress: bool = ...,
) -> list[str]: ...


def truncate_texts(
    texts: str | list[str],
    model: str | LLAMACPP_KEYS = LLM_MODEL,
    max_tokens: int | None = None,
    strict_sentences: bool = True,
    show_progress: bool = True,
) -> str | list[str]:
    """Truncate texts to a maximum token count, preserving sentence boundaries when possible.

    Based on text_chunker.py's truncate_texts_fast approach with these advantages:
    - Uses split_sentences_with_separators() for single-pass splitting with separators included
    - No manual separator extraction or sentence position hunting after splitting
    - No parallel sentence/separator array management
    - Simple "".join() reconstruction since separators are already part of sentences
    - Returns matching type: string input → string output, list input → list output

    Args:
        texts: Single text or list of texts to truncate.
        model: Model key for tokenizer (default: LLM_MODEL).
        max_tokens: Maximum tokens to keep. If None, uses model's context size.
        strict_sentences: If True, preserve sentence boundaries. If False, truncate at token level.
        show_progress: Show progress bar during batch processing.

    Returns:
        Truncated text string (if input was str) or list of truncated strings (if input was list).
        Empty strings from empty inputs are preserved for string input, filtered for list input.
    """
    from jet.wordnet.sentence import split_sentences_with_separators

    # Track if input was a single string for return type matching
    single_input = isinstance(texts, str)
    if single_input:
        texts = [texts]

    # Get max_tokens from model if not provided
    if max_tokens is None:
        try:
            from jet.adapters.llama_cpp.model_utils import get_model_ctx_embd_size

            ctx_info = get_model_ctx_embd_size(model)
            max_tokens = ctx_info["ctx"]
            logger.debug(f"Using model context size as max_tokens: {max_tokens}")
        except Exception as e:
            logger.warning(
                f"Could not get context size for {model}, using default 2048: {e}"
            )
            max_tokens = 2048

    tokenizer = get_tokenizer(model)
    results = []

    # Only show progress bar for batch processing (>1 item)
    if show_progress and len(texts) > 1:
        text_iter = tqdm(texts, desc="Truncating texts", unit="doc")
    else:
        text_iter = texts

    for text in text_iter:
        # Handle empty/whitespace-only text
        if not text or not text.strip():
            if single_input:
                results.append("")
            continue

        # Get original token count for logging
        original_tokens = len(tokenizer.encode(text, add_special_tokens=False))

        # Early exit: entire text fits within limit
        if original_tokens <= max_tokens:
            results.append(text.strip())
            logger.debug(
                f"Text fits within limit ({original_tokens}/{max_tokens} tokens), no truncation needed"
            )
            continue

        # Non-strict mode: simple token-level truncation
        if not strict_sentences:
            tokens = tokenizer.encode(text, add_special_tokens=False)[:max_tokens]
            truncated = tokenizer.decode(
                tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            ).strip()
            results.append(truncated)
            logger.debug(
                f"Token-level truncation: {original_tokens} → {max_tokens} tokens"
            )
            continue

        # Strict mode: preserve sentence boundaries
        # split_sentences_with_separators returns sentences with separators already attached
        sentences = split_sentences_with_separators(text)
        if not sentences:
            logger.debug(f"No sentences found in text: {text[:50]}...")
            if single_input:
                results.append("")
            continue

        current_tokens = 0
        kept_sentences = []
        total_sentences = len(sentences)

        for sentence in sentences:
            # Tokenize this sentence to get accurate token count
            sentence_tokens = tokenizer.encode(sentence, add_special_tokens=False)
            sentence_len = len(sentence_tokens)

            # Stop if adding this sentence would exceed the limit
            if current_tokens + sentence_len > max_tokens:
                logger.debug(
                    f"Truncating at sentence boundary: {current_tokens}/{max_tokens} tokens, "
                    f"next sentence adds {sentence_len} tokens "
                    f"(kept {len(kept_sentences)}/{total_sentences} sentences)"
                )
                break

            kept_sentences.append(sentence)
            current_tokens += sentence_len

        if kept_sentences:
            # Reconstruct by joining sentences directly (separators are included)
            truncated = "".join(kept_sentences).strip()
            results.append(truncated)
            logger.debug(
                f"Sentence-boundary truncation: {original_tokens} → {current_tokens} tokens, "
                f"{len(kept_sentences)}/{total_sentences} sentences kept"
            )
        else:
            # No complete sentence fits - fall back to token-level truncation of first sentence
            logger.debug(
                f"No complete sentence fits within {max_tokens} tokens, "
                f"falling back to token-level truncation of first sentence"
            )
            first_sentence = sentences[0]
            tokens = tokenizer.encode(first_sentence, add_special_tokens=False)[
                :max_tokens
            ]
            truncated = tokenizer.decode(
                tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            ).strip()
            results.append(truncated)

    # Log summary
    non_empty = sum(1 for r in results if r)
    logger.info(
        f"Truncation complete: {len(texts)} input(s) → {len(results)} output(s) "
        f"({non_empty} non-empty)"
    )

    # Return single string or list based on input type
    return results[0] if single_input else results


if __name__ == "__main__":
    from jet.adapters.llama_cpp.main._main_chunking_utils import main

    main()
