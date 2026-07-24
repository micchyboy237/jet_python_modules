import uuid
from typing import TypedDict

from jet.adapters.llama_cpp.token_utils import (
    detokenize,
    tokenize,
)
from jet.logger import logger
from jet.wordnet.sentence import (
    split_sentences,
)
from jet.wordnet.words import get_words
from tqdm import tqdm


def _tokenize_for_size(text, model: str | None = None):
    """Tokenize and return token IDs list for size counting."""
    if model is None:
        return get_words(text)
    result = tokenize(text, model=model)
    tokens = result["tokens"]
    if tokens and isinstance(tokens[0], dict):
        return [t["id"] for t in tokens]
    return tokens


def _get_size_fn(model: str | None = None):
    """Return a callable size_fn compatible with existing chunk_texts logic.

    Replaces get_tokenizer_fn() from jet._token.token_utils.
    Handles both single strings (returns list of token IDs) and
    lists of strings (returns list of list of token IDs).
    """
    if model is None:
        return get_words

    def _fn(text, show_progress=False):
        if isinstance(text, list):
            # Batch mode: return list of token ID lists
            return [_tokenize_for_size(t, model) for t in text]
        else:
            # Single string mode: return single list of token IDs
            return _tokenize_for_size(text, model)

    return _fn


def _decode_tokens(tokens: list[int], model: str | None = None) -> str:
    """Decode token IDs back to text. Replaces tokenizer.decode() calls."""
    if model is None:
        return " ".join(str(t) for t in tokens)
    result = detokenize(tokens, model=model)
    return result["content"]


def _encode_text(
    text: str, model: str | None = None, add_special: bool = False
) -> list[int]:
    """Encode text to token IDs. Replaces tokenizer.encode() calls."""
    if model is None:
        return get_words(text)
    result = tokenize(text, model=model, add_special=add_special)
    tokens = result["tokens"]
    if tokens and isinstance(tokens[0], dict):
        return [t["id"] for t in tokens]
    return tokens


def _get_last_n_tokens_and_decode(text: str, n: int, model: str | None = None) -> str:
    """Get the last n tokens from text and decode them back to string.

    Replaces get_last_n_tokens_and_decode() from jet._token.token_utils.
    """
    if model is None:
        words = get_words(text)
        return " ".join(words[-n:]) if n > 0 else ""
    token_ids = _encode_text(text, model)
    last_n = token_ids[-n:] if len(token_ids) >= n else token_ids
    return _decode_tokens(last_n, model)


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


def chunk_texts(
    texts: str | list[str],
    chunk_size: int = 128,
    chunk_overlap: int = 0,
    model: str | None = None,
    buffer: int = 0,
    strict_sentences: bool = True,
    min_chunk_size: int = 32,
    show_progress: bool = True,
) -> list[str]:
    """Optimized version of chunk_texts with O(n) time complexity per text."""
    if min_chunk_size > chunk_size:
        min_chunk_size = chunk_size
    if isinstance(texts, str):
        texts = [texts]
    chunked_texts = []
    size_fn = _get_size_fn(model) if model else get_words
    effective_chunk_size = chunk_size - buffer
    step = max(1, chunk_size - chunk_overlap - buffer)
    text_iter = tqdm(
        texts, desc="Chunking texts", unit="text", disable=not show_progress
    )
    for text in text_iter:
        sentences = split_sentences(text)
        if not sentences:
            continue
        if not strict_sentences and model:
            tokens = size_fn(text)
            total_len = len(tokens)
            if not tokens:
                continue
            for i in range(0, total_len, step):
                left, right = i, min(i + effective_chunk_size, total_len)
                chunk_tokens = []
                chunk_content = ""
                chunk_size_tokens = 0
                best_size = 0
                while left <= right:
                    mid = (left + right) // 2
                    temp_tokens = tokens[i:mid]
                    if temp_tokens:
                        temp_content = _decode_tokens(temp_tokens, model).strip()
                        temp_size = len(size_fn(temp_content))
                        if temp_size <= chunk_size:
                            if temp_size > best_size:
                                chunk_tokens = temp_tokens
                                chunk_content = temp_content
                                chunk_size_tokens = temp_size
                                best_size = temp_size
                            left = mid + 1
                        else:
                            right = mid - 1
                    else:
                        break
                if not chunk_tokens:
                    continue
                is_last_chunk = i + effective_chunk_size >= total_len
                if is_last_chunk and not chunk_tokens:
                    chunk_tokens = tokens[i:right]
                    if chunk_tokens:
                        chunk_content = _decode_tokens(chunk_tokens, model).strip()
                        chunk_size_tokens = len(size_fn(chunk_content))
                if not chunk_tokens:
                    continue
                if (
                    chunk_size_tokens < min_chunk_size
                    and not is_last_chunk
                    and chunk_size > min_chunk_size
                ):
                    continue
                chunked_texts.append(chunk_content)
            if (
                len(chunked_texts) > 1
                and len(size_fn(chunked_texts[-1])) < min_chunk_size
                and chunk_size > min_chunk_size
            ):
                last_chunk = chunked_texts.pop()
                prev_chunk = chunked_texts[-1]
                prev_chunk_last_n_tokens_string = _get_last_n_tokens_and_decode(
                    prev_chunk, len(size_fn(last_chunk)), model
                )
                is_covered_by_prev_chunk = last_chunk == prev_chunk_last_n_tokens_string
                if not is_covered_by_prev_chunk:
                    chunked_texts[-1] = prev_chunk + " " + last_chunk
            continue
        sent_sizes = [len(size_fn(s)) for s in sentences]
        i, n = 0, len(sentences)
        current_chunk, current_size = [], 0
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
                        overlap_tokens = []
                        overlap_len = 0
                        for sent in reversed(current_chunk):
                            overlap_len += len(size_fn(sent))
                            overlap_tokens.insert(0, sent)
                            if overlap_len >= chunk_overlap:
                                break
                        current_chunk = overlap_tokens
                        current_size = sum(len(size_fn(s)) for s in overlap_tokens)
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
    model: str | None = None,
    ids: list[str] | None = None,
    buffer: int = 0,
    strict_sentences: bool = True,
    min_chunk_size: int = 32,
    show_progress: bool = True,
) -> list[ChunkResult]:
    """Optimized version: removed binary search + repeated decoding in token path."""
    if min_chunk_size > chunk_size:
        min_chunk_size = chunk_size
    if isinstance(texts, str):
        texts = [texts]
        doc_indices = [0]
    else:
        doc_indices = list(range(len(texts)))
    chunks: list[ChunkResult] = []
    effective_chunk_size = chunk_size - buffer
    size_fn = _get_size_fn(model) if model else get_words
    batch_tokens = (
        size_fn(texts, show_progress=show_progress) if model else size_fn(texts)
    )
    token_counts = [len(tokens) for tokens in batch_tokens]
    step = max(1, chunk_size - chunk_overlap - buffer)
    logger.debug(
        f"chunk_texts_with_data vars: effective={effective_chunk_size}, "
        f"min_chunk={min_chunk_size}, step={step}, "
        f"chunk_size={chunk_size}, overlap={chunk_overlap}, buffer={buffer}, "
        f"docs={len(texts)}, avg_tokens={sum(token_counts) / len(token_counts):.0f}"
    )
    for i, (doc_index, text) in enumerate(
        tqdm(
            zip(doc_indices, texts),
            total=len(texts),
            desc="Chunking texts",
            disable=not show_progress,
        )
    ):
        if not text.strip():
            continue
        doc_id = ids[i] if ids and i < len(ids) else str(uuid.uuid4())
        if not strict_sentences and model:
            tokens = batch_tokens[i]
            if not tokens:
                continue
            total_len = len(tokens)
            chunk_index = 0
            j = 0
            while j < total_len:
                end = min(j + effective_chunk_size, total_len)
                chunk_tokens = tokens[j:end]
                if not chunk_tokens:
                    break
                chunk_content = _decode_tokens(chunk_tokens, model).strip()
                chunk_size_tokens = len(size_fn(chunk_content))
                is_last_chunk = end == total_len
                if (
                    chunk_size_tokens < min_chunk_size
                    and not is_last_chunk
                    and chunk_size > min_chunk_size
                ):
                    j += step
                    continue
                overlap_start_idx = overlap_end_idx = None
                if chunk_overlap > 0 and end < total_len:
                    overlap_start = max(j, end - chunk_overlap)
                    overlap_tokens = tokens[overlap_start:end]
                    if overlap_tokens:
                        overlap_start_idx = overlap_start
                        overlap_end_idx = end
                chunks.append(
                    {
                        "id": str(uuid.uuid4()),
                        "doc_id": doc_id,
                        "doc_index": doc_index,
                        "chunk_index": chunk_index,
                        "num_tokens": chunk_size_tokens,
                        "content": chunk_content,
                        "start_idx": j,
                        "end_idx": j + len(chunk_content),
                        "line_idx": 0,
                        "overlap_start_idx": overlap_start_idx,
                        "overlap_end_idx": overlap_end_idx,
                    }
                )
                chunk_index += 1
                j += step
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
        sentences = split_sentences(text)
        if not sentences:
            continue
        sent_sizes = [len(size_fn(s)) for s in sentences]
        current_chunk: list[str] = []
        current_size = 0
        chunk_index = 0
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


if __name__ == "__main__":
    from jet.adapters.llama_cpp.main._main_chunking_utils import main

    main()
