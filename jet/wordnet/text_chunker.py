import uuid
import re
from nltk.tokenize import sent_tokenize
from typing import TypedDict, Union, List, Tuple, Optional
from tqdm import tqdm
from jet.code.markdown_types.markdown_parsed_types import MarkdownToken
from jet.llm.models import OLLAMA_MODEL_NAMES
from jet.logger import logger
# from jet.vectors.document_types import HeaderDocument, HeaderMetadata
from jet._token.token_utils import get_last_n_tokens_and_decode, get_model_max_tokens, get_tokenizer, get_tokenizer_fn
from jet.wordnet.sentence import split_sentences, is_list_marker, is_list_sentence, split_sentences_with_separators
from jet.wordnet.utils import sliding_window
from jet.wordnet.words import get_words


def build_chunk(sentences: List[str], separators: List[str]) -> str:
    """Reconstruct a chunk from sentences and separators, removing leading/trailing whitespaces."""
    chunk = ""
    for sentence, separator in zip(sentences, separators):
        chunk += sentence + separator
    return chunk.strip()


def get_overlap_sentences(
    sentences: List[str],
    separators: List[str],
    max_overlap: int,
    size_fn,
) -> Tuple[List[str], List[str], int]:
    """Select sentences for overlap based on size (tokens or words)."""
    if len(sentences) != len(separators):
        raise IndexError(
            f"Sentences ({len(sentences)}) and separators ({len(separators)}) lists must have the same length")
    overlap_sentences = []
    overlap_separators = []
    overlap_size = 0
    for sentence, separator in reversed(list(zip(sentences, separators))):
        sentence_size = len(size_fn(sentence))
        if overlap_size + sentence_size <= max_overlap:
            overlap_sentences.insert(0, sentence)
            overlap_separators.insert(0, separator)
            overlap_size += sentence_size
        else:
            break
    return overlap_sentences, overlap_separators, overlap_size


def split_large_sentence(sentence: str, max_size: int, size_fn) -> List[str]:
    """Split a large sentence into smaller parts based on max_size (tokens)."""
    sentence_tokens = len(size_fn(sentence))
    if sentence_tokens <= max_size:
        return [sentence]

    # logger.warning(
    #     f"Splitting sentence with {sentence_tokens} tokens exceeding max_size {max_size}")
    words = get_words(sentence)
    sub_sentences = []
    current_sub = []
    current_size = 0

    for word in words:
        word_size = len(size_fn(word))
        if current_size + word_size <= max_size:
            current_sub.append(word)
            current_size += word_size
        else:
            if current_sub:
                sub_sentence = " ".join(current_sub) + "."
                # Verify sub-sentence size
                if len(size_fn(sub_sentence)) <= max_size:
                    sub_sentences.append(sub_sentence)
                else:
                    # logger.warning(
                    #     f"Sub-sentence '{sub_sentence}' still exceeds max_size {max_size}")
                    # Fallback: split by character length
                    chars = " ".join(current_sub)
                    while chars:
                        # Rough estimate: 4 chars per token
                        sub_sentence = chars[:max_size * 4] + "."
                        sub_sentences.append(sub_sentence)
                        chars = chars[max_size * 4:]
                current_sub = [word]
                current_size = word_size
            else:
                # Single word exceeds max_size
                sub_sentences.append(word + ".")
                # logger.warning(
                #     f"Single word '{word}' exceeds max_size {max_size}")
                current_sub = []
                current_size = 0

    if current_sub:
        sub_sentence = " ".join(current_sub) + "."
        if len(size_fn(sub_sentence)) <= max_size:
            sub_sentences.append(sub_sentence)
        else:
            chars = " ".join(current_sub)
            while chars:
                sub_sentences.append(chars[:max_size * 4] + ".")
                chars = chars[max_size * 4:]

    return sub_sentences


def normalize_separator(separator: str, max_length: int = 10) -> str:
    """Preserve original separator, truncating if too long."""
    if len(separator) > max_length:
        # logger.warning(
        #     f"Truncating separator of {len(separator)} characters to {max_length}")
        return separator[:max_length]
    return separator if separator else " "


class ChunkResultMeta(TypedDict):
    """Metadata for document chunks.

    Attributes:
        doc_id: Document ID (same for all chunks of a document).
        doc_index: Document index in the source dataset.
        header: Header text (e.g., '### Title').
        level: Header level (e.g., 2 → '##').
        parent_header: Parent section header (e.g., '## Parent').
        parent_level: Parent header level (e.g., 2 → '##').
        source: File path, URL, or other source reference.
        tokens: List of parsed markdown tokens for this chunk.
    """

    doc_id: str
    doc_index: int
    header: str
    level: Optional[int]
    parent_header: Optional[str]
    parent_level: Optional[int]
    source: Optional[str]
    tokens: List[MarkdownToken]


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
    overlap_start_idx: Optional[int]
    overlap_end_idx: Optional[int]


class ChunkResultWithMeta(ChunkResult):
    """Chunk data extended with metadata.

    Attributes:
        meta: Metadata containing headers, structure, and source info.
    """

    meta: ChunkResultMeta


def chunk_texts(
    texts: Union[str, List[str]],
    chunk_size: int = 128,
    chunk_overlap: int = 0,
    model: Optional[Union[str, OLLAMA_MODEL_NAMES]] = None,
    buffer: int = 0,
    strict_sentences: bool = False,
    min_chunk_size: int = 32,
    show_progress: bool = False,
) -> List[str]:
    """Optimized version of chunk_texts with O(n) time complexity per text."""
    
    if min_chunk_size > chunk_size:
        min_chunk_size = chunk_size
    if isinstance(texts, str):
        texts = [texts]

    chunked_texts = []
    size_fn = get_tokenizer_fn(model) if model else get_words
    tokenizer = get_tokenizer(model) if model else None
    effective_chunk_size = chunk_size - buffer
    step = max(1, chunk_size - chunk_overlap - buffer)

    text_iter = tqdm(texts, desc="Chunking texts", unit="text", disable=not show_progress)
    
    for text in text_iter:
        # Fast split: use regex instead of tokenizer.find
        sentences = split_sentences(text)
        if not sentences:
            continue

        # Fast path: if model is used and not strict → chunk by tokens directly
        if not strict_sentences and model:
            tokens = size_fn(text)
            total_len = len(tokens)
            if not tokens:
                continue
            for i in range(0, total_len, step):
                # Binary search to find the largest slice <= chunk_size
                left, right = i, min(i + effective_chunk_size, total_len)
                chunk_tokens = []
                chunk_content = ""
                chunk_size_tokens = 0
                best_size = 0
                while left <= right:
                    mid = (left + right) // 2
                    temp_tokens = tokens[i:mid]
                    if temp_tokens:
                        temp_content = tokenizer.decode(temp_tokens).strip()
                        temp_size = len(size_fn(temp_content))
                        if temp_size <= chunk_size:
                            # Keep track of the best valid slice
                            if temp_size > best_size:
                                chunk_tokens = temp_tokens
                                chunk_content = temp_content
                                chunk_size_tokens = temp_size
                                best_size = temp_size
                            left = mid + 1  # Try a larger slice
                        else:
                            right = mid - 1  # Try a smaller slice
                    else:
                        break
                if not chunk_tokens:
                    continue
                # Handle last chunk: include remaining tokens if needed
                is_last_chunk = i + effective_chunk_size >= total_len
                if is_last_chunk and not chunk_tokens:
                    chunk_tokens = tokens[i:right]
                    if chunk_tokens:
                        chunk_content = tokenizer.decode(chunk_tokens).strip()
                        chunk_size_tokens = len(size_fn(chunk_content))
                if not chunk_tokens:
                    continue
                # Skip chunks smaller than min_chunk_size if not the last chunk
                if chunk_size_tokens < min_chunk_size and not is_last_chunk and chunk_size > min_chunk_size:
                    continue
                chunked_texts.append(chunk_content)
            # Merge last small chunk
            if len(chunked_texts) > 1 and len(size_fn(chunked_texts[-1])) < min_chunk_size and chunk_size > min_chunk_size:
                last_chunk = chunked_texts.pop()
                prev_chunk = chunked_texts[-1]
                prev_chunk_last_n_tokens_string = get_last_n_tokens_and_decode(prev_chunk, tokenizer, len(size_fn(last_chunk)))
                is_covered_by_prev_chunk = last_chunk == prev_chunk_last_n_tokens_string
                if not is_covered_by_prev_chunk:
                    chunked_texts[-1] = prev_chunk + " " + last_chunk
            continue

        # Sentence-based chunking (faster with precomputed sizes)
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
                    # Apply overlap
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
    texts: Union[str, List[str]],
    chunk_size: int = 128,
    chunk_overlap: int = 0,
    model: Optional[Union[str, OLLAMA_MODEL_NAMES]] = None,
    ids: Optional[List[str]] = None,
    buffer: int = 0,
    strict_sentences: bool = False,
    min_chunk_size: int = 32,
    show_progress: bool = True
) -> List[ChunkResult]:
    """Optimized, faster version of chunk_texts_with_data."""
    if min_chunk_size > chunk_size:
        min_chunk_size = chunk_size
    if isinstance(texts, str):
        texts = [texts]
        doc_indices = [0]
    else:
        doc_indices = list(range(len(texts)))

    chunks: List[ChunkResult] = []
    effective_chunk_size = chunk_size - buffer
    size_fn = get_tokenizer_fn(model) if model else get_words
    tokenizer = get_tokenizer(model) if model else None
    step = max(1, chunk_size - chunk_overlap - buffer)

    for i, (doc_index, text) in enumerate(tqdm(zip(doc_indices, texts), total=len(texts), desc="Chunking texts", disable=not show_progress)):
        sentences = split_sentences(text)
        if not sentences:
            continue
        doc_id = ids[i] if ids and i < len(ids) else str(uuid.uuid4())

        # Fast token-based path
        if not strict_sentences and model:
            tokens = size_fn(text)
            if not tokens:
                continue
            total_len = len(tokens)
            chunk_index = 0  # Initialize chunk_index for this document
            for j in range(0, total_len, step):
                # Binary search to find the largest slice <= chunk_size
                left, right = j, min(j + effective_chunk_size, total_len)
                chunk_tokens = []
                chunk_content = ""
                chunk_size_tokens = 0
                best_size = 0
                while left <= right:
                    mid = (left + right) // 2
                    temp_tokens = tokens[j:mid]
                    if temp_tokens:
                        temp_content = tokenizer.decode(temp_tokens).strip()
                        temp_size = len(size_fn(temp_content))
                        if temp_size <= chunk_size:
                            # Keep track of the best valid slice
                            if temp_size > best_size:
                                chunk_tokens = temp_tokens
                                chunk_content = temp_content
                                chunk_size_tokens = temp_size
                                best_size = temp_size
                            left = mid + 1  # Try a larger slice
                        else:
                            right = mid - 1  # Try a smaller slice
                    else:
                        break
                if not chunk_tokens:
                    continue
                # Handle last chunk: include remaining tokens if needed
                is_last_chunk = j + effective_chunk_size >= total_len
                if is_last_chunk and not chunk_tokens:
                    chunk_tokens = tokens[j:right]
                    if chunk_tokens:
                        chunk_content = tokenizer.decode(chunk_tokens).strip()
                        chunk_size_tokens = len(size_fn(chunk_content))
                if not chunk_tokens:
                    continue
                # Skip min_chunk_size check for last chunk or when chunk_size <= min_chunk_size
                if chunk_size_tokens < min_chunk_size and not is_last_chunk and chunk_size > min_chunk_size:
                    continue
                # Calculate overlap indices
                overlap_start_idx = None
                overlap_end_idx = None
                if chunk_overlap > 0 and j + len(chunk_tokens) < total_len:
                    overlap_start = j + len(chunk_tokens) - min(chunk_overlap, len(chunk_tokens))
                    overlap_end = min(j + len(chunk_tokens), total_len)
                    overlap_tokens = tokens[overlap_start:overlap_end]
                    if overlap_tokens:
                        overlap_start_idx = overlap_start
                        overlap_end_idx = overlap_end
                chunks.append({
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
                    "overlap_end_idx": overlap_end_idx
                })
                chunk_index += 1
            # Merge last too-small chunk
            if len(chunks) > 1 and chunks[-1]["num_tokens"] < min_chunk_size and chunk_size > min_chunk_size:
                last = chunks.pop()
                prev = chunks[-1]
                prev_last_n_tokens_string = get_last_n_tokens_and_decode(prev["content"], tokenizer, last["num_tokens"])
                is_covered_by_prev_chunk = last["content"] == prev_last_n_tokens_string
                if not is_covered_by_prev_chunk:
                    prev["content"] += " " + last["content"]
                    prev["num_tokens"] = len(size_fn(prev["content"]))
                    prev["end_idx"] = last["end_idx"]
            continue
        # Sentence-based path
        sent_sizes = [len(size_fn(s)) for s in sentences]
        current_chunk, current_size = [], 0
        chunk_index = 0
        for s, s_size in zip(sentences, sent_sizes):
            if s_size > effective_chunk_size:
                sub_sents = split_large_sentence(s, effective_chunk_size, size_fn)
                for sub in sub_sents:
                    sub_size = len(size_fn(sub))
                    if current_size + sub_size > effective_chunk_size:
                        if current_chunk:
                            chunk_content = " ".join(current_chunk)
                            num_tokens = len(size_fn(chunk_content))
                            if num_tokens >= min_chunk_size or not chunks:
                                chunks.append({
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
                                    "overlap_end_idx": None
                                })
                                chunk_index += 1
                        current_chunk, current_size = [], 0
                    current_chunk.append(sub)
                    current_size += sub_size
            else:
                if current_size + s_size > effective_chunk_size and current_chunk:
                    chunk_content = " ".join(current_chunk)
                    num_tokens = len(size_fn(chunk_content))
                    if num_tokens >= min_chunk_size or not chunks:
                        chunks.append({
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
                            "overlap_end_idx": None
                        })
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
                chunks.append({
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
                    "overlap_end_idx": None
                })
    return chunks


def chunk_texts_sliding_window(
    texts: Union[str, List[str]],
    chunk_size: int = 128,
    step_size: int = 96,
    model: Optional[Union[str, OLLAMA_MODEL_NAMES]] = None,
    ids: Optional[List[str]] = None,
    buffer: int = 0,
    min_chunk_size: int = 32,
) -> List[ChunkResult]:
    """
    Chunk texts using a sliding window approach, returning detailed chunk results.
    
    Args:
        texts: Single string or list of strings to chunk.
        chunk_size: Number of tokens per chunk.
        step_size: Number of tokens to move the window at each step.
        model: Optional LLM model name for token-based chunking.
        ids: Optional list of document IDs.
        buffer: Buffer size to reserve in each chunk.
        min_chunk_size: Minimum number of tokens for a chunk to be valid.
    
    Returns:
        List of ChunkResult dictionaries containing chunk metadata.
    
    Raises:
        ValueError: If step_size <= 0 or step_size >= chunk_size.
    """
    if step_size <= 0 or step_size >= chunk_size:
        raise ValueError("step_size must be greater than 0 and less than chunk_size")
    if min_chunk_size > chunk_size:
        min_chunk_size = chunk_size
    if isinstance(texts, str):
        texts = [texts]
        doc_indices = [0] * len(texts)
    else:
        doc_indices = list(range(len(texts)))
    
    chunks: List[ChunkResult] = []
    size_fn = get_tokenizer_fn(model) if model else get_words
    tokenizer = get_tokenizer(model) if model else None
    step = step_size
    
    for i, (doc_index, text) in enumerate(zip(doc_indices, texts)):
        tokens = size_fn(text)
        if not tokens:
            continue
        doc_id = ids[i] if ids and i < len(ids) else str(uuid.uuid4())
        windows = sliding_window(tokens, chunk_size - buffer, step_size)
        
        for chunk_index, window in enumerate(windows):
            chunk_content = tokenizer.decode(window).strip() if model else " ".join(window).strip()
            chunk_size_tokens = len(size_fn(chunk_content))
            
            if chunk_size_tokens < min_chunk_size:
                continue
                
            start_idx = chunk_index * step_size
            end_idx = start_idx + len(chunk_content)
            
            chunks.append({
                "id": str(uuid.uuid4()),
                "doc_id": doc_id,
                "doc_index": doc_index,
                "chunk_index": chunk_index,
                "num_tokens": chunk_size_tokens,
                "content": chunk_content,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "line_idx": 0,
                "overlap_start_idx": None,
                "overlap_end_idx": None
            })
    
    # Merge small last chunk with previous if necessary
    if len(chunks) > 1 and chunks[-1]["num_tokens"] < min_chunk_size:
        last_chunk = chunks.pop()
        prev_chunk = chunks[-1]
        prev_chunk_last_n_tokens_string = get_last_n_tokens_and_decode(
            prev_chunk["content"], tokenizer, last_chunk["num_tokens"]
        )
        is_covered_by_prev_chunk = last_chunk["content"] == prev_chunk_last_n_tokens_string
        if not is_covered_by_prev_chunk:
            prev_chunk["content"] += " " + last_chunk["content"]
            prev_chunk["num_tokens"] = len(size_fn(prev_chunk["content"]))
            prev_chunk["end_idx"] = last_chunk["end_idx"]
    
    return chunks


def chunk_texts_sliding_window_fast(
    texts: Union[str, List[str]],
    chunk_size: int = 128,
    step_size: int = 96,
    model: Optional[Union[str, OLLAMA_MODEL_NAMES]] = None,
    ids: Optional[List[str]] = None,
    buffer: int = 0,
    min_chunk_size: int = 32,
    show_progress: bool = True
) -> List[ChunkResult]:
    """Optimized sliding-window chunking with minimal recomputation."""
    if step_size <= 0 or step_size >= chunk_size:
        raise ValueError("step_size must be >0 and <chunk_size")
    if min_chunk_size > chunk_size:
        min_chunk_size = chunk_size
    if isinstance(texts, str):
        texts = [texts]
        doc_indices = [0]
    else:
        doc_indices = list(range(len(texts)))

    chunks: List[ChunkResult] = []
    size_fn = get_tokenizer_fn(model) if model else get_words
    tokenizer = get_tokenizer(model) if model else None
    effective_size = chunk_size - buffer

    for i, (doc_index, text) in enumerate(tqdm(zip(doc_indices, texts), total=len(texts), desc="Sliding window", disable=not show_progress)):
        tokens = size_fn(text)
        if not tokens:
            continue
        doc_id = ids[i] if ids and i < len(ids) else str(uuid.uuid4())

        for chunk_index in range(0, len(tokens) - effective_size + 1, step_size):
            window = tokens[chunk_index:chunk_index + effective_size]
            chunk_content = tokenizer.decode(window).strip() if model else " ".join(window).strip()
            num_tokens = len(window)
            if num_tokens < min_chunk_size:
                continue
            chunks.append({
                "id": str(uuid.uuid4()),
                "doc_id": doc_id,
                "doc_index": doc_index,
                "chunk_index": chunk_index // step_size,
                "num_tokens": num_tokens,
                "content": chunk_content,
                "start_idx": chunk_index,
                "end_idx": chunk_index + len(chunk_content),
                "line_idx": 0,
                "overlap_start_idx": None,
                "overlap_end_idx": None
            })

    # Merge small tail
    if len(chunks) > 1 and chunks[-1]["num_tokens"] < min_chunk_size:
        last = chunks.pop()
        prev = chunks[-1]
        prev["content"] += " " + last["content"]
        prev["num_tokens"] = len(size_fn(prev["content"]))
        prev["end_idx"] = last["end_idx"]

    return chunks


def truncate_texts(
    texts: str | List[str],
    model: Union[str, OLLAMA_MODEL_NAMES],
    max_tokens: Optional[int] = None,
    strict_sentences: bool = False
) -> List[str]:
    tokenizer = get_tokenizer(model)
    if not max_tokens:
        max_tokens = get_model_max_tokens(model)
    if isinstance(texts, str):
        texts = [texts]
    truncated_texts = []
    for text in texts:
        if not strict_sentences:
            # Truncate based on tokens without sentence boundaries
            tokens = tokenizer.encode(text, add_special_tokens=False)[:max_tokens]
            truncated_texts.append(tokenizer.decode(tokens).strip())
            continue
        sentences = split_sentences(text)
        if not sentences:
            truncated_texts.append("")
            continue
        sentence_pairs = []
        current_pos = 0
        i = 0
        while i < len(sentences):
            current_sentence = sentences[i]
            start_idx = text.find(current_sentence, current_pos)
            if start_idx == -1:
                sentence_pairs.append((current_sentence, " "))
                i += 1
                continue
            end_idx = start_idx + len(current_sentence)
            separator = text[end_idx:text.find(
                sentences[i + 1], end_idx) if i + 1 < len(sentences) else len(text)]
            separator = normalize_separator(separator)
            if is_list_marker(current_sentence) and i + 1 < len(sentences):
                combined = current_sentence + " " + sentences[i + 1]
                if is_list_sentence(combined):
                    combined_start = text.find(combined, current_pos)
                    if combined_start != -1:
                        combined_end = combined_start + len(combined)
                        combined_separator = text[combined_end:text.find(
                            sentences[i + 2], combined_end) if i + 2 < len(sentences) else len(text)]
                        combined_separator = normalize_separator(combined_separator)
                        sentence_pairs.append((combined, combined_separator))
                        current_pos = combined_end
                        i += 2
                        continue
            sentence_pairs.append((current_sentence, separator))
            current_pos = end_idx
            i += 1
        current_chunk = []
        current_separators = []
        current_tokens = 0
        for sentence, separator in sentence_pairs:
            sentence_tokens = len(tokenizer.encode(sentence, add_special_tokens=False))
            if current_tokens + sentence_tokens <= max_tokens:
                current_chunk.append(sentence)
                current_separators.append(separator)
                current_tokens += sentence_tokens
            else:
                break
        if current_chunk:
            chunk = build_chunk(current_chunk, current_separators)
            truncated_texts.append(chunk)
        else:
            truncated_texts.append("")
    return truncated_texts


def truncate_texts_fast(
    texts: Union[str, List[str]],
    model: Union[str, OLLAMA_MODEL_NAMES],
    max_tokens: Optional[int] = None,
    strict_sentences: bool = False,
    show_progress: bool = False
) -> List[str]:
    """Optimized version of truncate_texts (up to 10× faster) with optional progress bar."""
    tokenizer = get_tokenizer(model)
    if not max_tokens:
        max_tokens = get_model_max_tokens(model)
    if isinstance(texts, str):
        texts = [texts]

    iterator = tqdm(texts, desc="Truncating texts", unit="doc") if show_progress else texts
    results = []

    for text in iterator:
        if not strict_sentences:
            tokens = tokenizer.encode(text, add_special_tokens=False)[:max_tokens]
            results.append(tokenizer.decode(tokens).strip())
            continue

        sentences = split_sentences_with_separators(text)
        if not sentences:
            results.append("")
            continue

        current_tokens = 0
        kept_sentences = []
        for s in sentences:
            s_len = len(tokenizer.encode(s, add_special_tokens=False))
            if current_tokens + s_len > max_tokens:
                break
            kept_sentences.append(s)
            current_tokens += s_len

        results.append("".join(kept_sentences).strip())

    results = [r for r in results if r]
    return results


def chunk_sentences(texts: Union[str, List[str]], chunk_size: int = 5, chunk_overlap: int = 0, model: Optional[Union[str, OLLAMA_MODEL_NAMES]] = None) -> List[str]:
    """Chunk texts by sentences with sentence overlap, using tokens if model is provided, preserving original separators.

    Args:
        texts: Single string or list of strings to chunk.
        chunk_size: Number of sentences (non-model) or tokens (model) per chunk.
        chunk_overlap: Number of sentences to overlap.
        model: Optional LLM model name for token-based chunking.
    """
    if isinstance(texts, str):
        texts = [texts]
    chunked_texts = []

    if model:
        tokenize_fn = get_tokenizer_fn(model)
        for text in texts:
            sentence_pairs = split_sentences_with_separator_tuples(text)
            sentences = [s for s, _ in sentence_pairs]
            separators = [sep for _, sep in sentence_pairs]
            if not sentences:
                # logger.debug(f"No sentences found for text: {text}")
                continue
            # Check total tokens against chunk_size
            total_tokens = sum(len(tokenize_fn(s)) for s in sentences)
            logger.debug(
                f"Total tokens: {total_tokens}, chunk_size: {chunk_size}")
            if total_tokens <= chunk_size:
                logger.debug(f"Text too short, appending original: {text}")
                chunked_texts.append(text)
                continue
            current_chunk = []
            current_separators = []
            current_tokens = 0
            for i, sentence in enumerate(sentences):
                sentence_tokens = len(tokenize_fn(sentence))
                logger.debug(
                    f"Sentence {i}: '{sentence}', tokens: {sentence_tokens}, current_tokens: {current_tokens}")
                if current_tokens + sentence_tokens > chunk_size and current_chunk:
                    # Reconstruct chunk with original separators
                    chunk = ""
                    for j in range(len(current_chunk)):
                        chunk += current_chunk[j]
                        if j < len(current_chunk) - 1:
                            chunk += current_separators[j] if j < len(
                                current_separators) else " "
                    logger.debug(
                        f"Appending chunk: '{chunk}', tokens: {current_tokens}")
                    chunked_texts.append(chunk)
                    current_chunk = []
                    current_separators = []
                    current_tokens = 0
                    # Handle overlap (in sentences, not tokens)
                    overlap_start = max(0, i - chunk_overlap)
                    logger.debug(
                        f"Overlap start: {overlap_start}, current index: {i}")
                    current_chunk = sentences[overlap_start:i]
                    current_separators = separators[overlap_start:i] if overlap_start < len(
                        separators) else []
                    current_tokens = sum(len(tokenize_fn(s))
                                         for s in current_chunk)
                    logger.debug(
                        f"New chunk after overlap: {current_chunk}, tokens: {current_tokens}")
                current_chunk.append(sentence)
                current_separators.append(
                    separators[i] if i < len(separators) else " ")
                current_tokens += sentence_tokens
            if current_chunk:
                # Reconstruct final chunk with original separators
                chunk = ""
                for j in range(len(current_chunk)):
                    chunk += current_chunk[j]
                    if j < len(current_chunk) - 1:
                        chunk += current_separators[j] if j < len(
                            current_separators) else " "
                logger.debug(
                    f"Appending final chunk: '{chunk}', tokens: {current_tokens}")
                chunked_texts.append(chunk)
    else:
        for text in texts:
            sentence_pairs = split_sentences_with_separator_tuples(text)
            sentences = [s for s, _ in sentence_pairs]
            separators = [sep for _, sep in sentence_pairs]
            sentences = [s.strip() for s in sentences if s.strip()]
            if not sentences:
                continue
            if len(sentences) > chunk_size:
                for i in range(0, len(sentences) - chunk_size + 1, chunk_size - chunk_overlap):
                    start_idx = max(0, i)
                    end_idx = min(len(sentences), i + chunk_size)
                    # Reconstruct chunk with original separators
                    chunk = ""
                    for j in range(start_idx, end_idx):
                        chunk += sentences[j]
                        if j < end_idx - 1:  # Add separator except for the last sentence
                            chunk += separators[j] if j < len(
                                separators) else " "
                    chunked_texts.append(chunk)
            else:
                chunked_texts.append(text)
    return chunked_texts


def chunk_sentences_optimized(
    texts: Union[str, List[str]],
    chunk_size: int = 5,
    chunk_overlap: int = 0,
    model: Optional[Union[str, OLLAMA_MODEL_NAMES]] = None,
    show_progress: bool = False
) -> List[str]:
    """Fast, memory-efficient sentence chunking with optional progress bar."""
    if isinstance(texts, str):
        texts = [texts]

    chunked_texts = []
    iterator = tqdm(texts, desc="Chunking sentences", unit="doc") if show_progress else texts

    if model:
        tokenize_fn = get_tokenizer_fn(model)
        encode_cache = {}

        for text in iterator:
            sentence_pairs = split_sentences_with_separator_tuples(text)
            if not sentence_pairs:
                continue

            sentences, separators = zip(*sentence_pairs)
            token_lengths = [
                encode_cache.setdefault(s, len(tokenize_fn(s))) for s in sentences
            ]
            total_tokens = sum(token_lengths)

            if total_tokens <= chunk_size:
                chunked_texts.append(text)
                continue

            start = 0
            while start < len(sentences):
                current_tokens, end = 0, start
                while end < len(sentences) and current_tokens + token_lengths[end] <= chunk_size:
                    current_tokens += token_lengths[end]
                    end += 1
                if end == start:
                    end += 1

                chunk = "".join(
                    s + (separators[i] if i < len(separators) else " ")
                    for i, s in enumerate(sentences[start:end])
                ).strip()
                chunked_texts.append(chunk)

                if end >= len(sentences):
                    break
                start = max(0, end - chunk_overlap)
    else:
        for text in iterator:
            sentence_pairs = split_sentences_with_separator_tuples(text)
            if not sentence_pairs:
                continue

            sentences, separators = zip(*sentence_pairs)
            sentences = [s.strip() for s in sentences if s.strip()]
            if not sentences:
                continue

            n = len(sentences)
            if n <= chunk_size:
                chunked_texts.append(text)
                continue

            step = chunk_size - chunk_overlap
            for i in range(0, n, step):
                end = min(i + chunk_size, n)
                chunk = "".join(
                    s + (separators[j] if j < len(separators) else " ")
                    for j, s in enumerate(sentences[i:end])
                ).strip()
                chunked_texts.append(chunk)

    return chunked_texts


def chunk_sentences_with_indices(texts: Union[str, List[str]], chunk_size: int = 5, chunk_overlap: int = 0, model: Optional[Union[str, OLLAMA_MODEL_NAMES]] = None) -> Tuple[List[str], List[int]]:
    """Chunk texts by sentences with sentence overlap and track original document indices, using tokens if model is provided.

    Args:
        texts: Single string or list of strings to chunk.
        chunk_size: Number of sentences or tokens per chunk.
        chunk_overlap: Number of sentences or tokens to overlap.
        model: Optional LLM model name for token-based chunking.
    """
    if isinstance(texts, str):
        texts = [texts]

    chunked_texts = []
    doc_indices = []
    sentence_splitter = re.compile(r'(?<=[.!?])\s+(?=\w)')

    if model:
        tokenize_fn = get_tokenizer_fn(model)
        for doc_idx, text in enumerate(texts):
            sentences = split_sentences(text.strip())
            current_chunk = []
            current_tokens = 0
            for i, sentence in enumerate(sentences):
                sentence_tokens = len(tokenize_fn(sentence))
                if current_tokens + sentence_tokens > chunk_size and current_chunk:
                    chunked_texts.append(" ".join(current_chunk))
                    doc_indices.append(doc_idx)
                    current_chunk = []
                    current_tokens = 0
                    # Handle overlap
                    overlap_start = max(0, i - chunk_overlap)
                    current_chunk = sentences[overlap_start:i]
                    current_tokens = sum(len(tokenize_fn(s))
                                         for s in current_chunk)
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
            if current_chunk:
                chunked_texts.append(" ".join(current_chunk))
                doc_indices.append(doc_idx)
    else:
        for doc_idx, text in enumerate(texts):
            sentences = sentence_splitter.split(text.strip())
            sentences = [s.strip() for s in sentences if s.strip()]
            if len(sentences) > chunk_size:
                for i in range(0, len(sentences) - chunk_size + 1, chunk_size - chunk_overlap):
                    start_idx = max(0, i)
                    end_idx = min(len(sentences), i + chunk_size)
                    chunked_texts.append(
                        " ".join(sentences[start_idx:end_idx]))
                    doc_indices.append(doc_idx)
            else:
                chunked_texts.append(text)
                doc_indices.append(doc_idx)
    return chunked_texts, doc_indices


def chunk_sentences_with_indices_optimized(
    texts: Union[str, List[str]],
    chunk_size: int = 5,
    chunk_overlap: int = 0,
    model: Optional[Union[str, OLLAMA_MODEL_NAMES]] = None,
    show_progress: bool = False
) -> Tuple[List[str], List[int]]:
    """Optimized sentence chunking with document index tracking and optional progress bar."""
    if isinstance(texts, str):
        texts = [texts]

    chunked_texts, doc_indices = [], []
    iterator = tqdm(enumerate(texts), desc="Chunking with indices", total=len(texts), unit="doc") if show_progress else enumerate(texts)

    if model:
        tokenize_fn = get_tokenizer_fn(model)
        encode_cache = {}

        for doc_idx, text in iterator:
            sentences = split_sentences(text.strip())
            if not sentences:
                continue

            token_lengths = [
                encode_cache.setdefault(s, len(tokenize_fn(s))) for s in sentences
            ]
            n = len(sentences)
            step = max(1, chunk_size - chunk_overlap)
            i = 0

            while i < n:
                current_tokens, end = 0, i
                while end < n and current_tokens + token_lengths[end] <= chunk_size:
                    current_tokens += token_lengths[end]
                    end += 1
                if end == i:
                    end += 1

                chunked_texts.append(" ".join(sentences[i:end]).strip())
                doc_indices.append(doc_idx)

                if end >= n:
                    break
                i = max(0, end - chunk_overlap)
    else:
        sentence_splitter = re.compile(r'(?<=[.!?])\s+(?=\w)')
        for doc_idx, text in iterator:
            sentences = [s.strip() for s in sentence_splitter.split(text.strip()) if s.strip()]
            if not sentences:
                continue

            n = len(sentences)
            if n <= chunk_size:
                chunked_texts.append(text)
                doc_indices.append(doc_idx)
                continue

            step = chunk_size - chunk_overlap
            for i in range(0, n, step):
                end = min(i + chunk_size, n)
                chunked_texts.append(" ".join(sentences[i:end]))
                doc_indices.append(doc_idx)

    return chunked_texts, doc_indices


# def chunk_headers(docs: List[HeaderDocument], max_tokens: int = 500, model: Optional[Union[str, OLLAMA_MODEL_NAMES]] = None) -> List[HeaderDocument]:
#     """Chunk HeaderDocument list into smaller segments based on token count or lines, ensuring complete sentences when model is provided.

#     Args:
#         docs: List of HeaderDocument objects to chunk.
#         max_tokens: Maximum number of tokens or lines per chunk.
#         model: Optional LLM model name for token-based chunking.
#     """
#     logger.debug("Starting chunk_headers with %d documents", len(docs))
#     chunked_docs: List[HeaderDocument] = []

#     for doc in docs:
#         chunk_index = 0
#         metadata = HeaderMetadata(**doc.metadata)
#         parent_header = metadata.get("parent_header", "")
#         doc_index = metadata.get("doc_index", 0)
#         # Use original header from metadata
#         header = metadata.get("header", "")

#         if model:
#             # Token-based chunking with sentence boundaries
#             tokenize_fn = get_tokenizer_fn(model)
#             sentences = split_sentences(doc.text)
#             current_chunk = []
#             current_tokens = 0
#             for sentence in sentences:
#                 sentence_tokens = len(tokenize_fn(sentence))
#                 if current_tokens + sentence_tokens > max_tokens and current_chunk:
#                     chunk_text = " ".join(current_chunk)
#                     chunked_docs.append(HeaderDocument(
#                         id=f"{doc.id}_chunk_{chunk_index}",
#                         text=chunk_text,
#                         metadata={
#                             "source_url": metadata.get("source_url", None),
#                             "header": header,
#                             "parent_header": parent_header,
#                             "header_level": metadata.get("header_level", 0) + 1,
#                             "content": chunk_text,
#                             "doc_index": doc_index,
#                             "chunk_index": chunk_index,
#                             "texts": current_chunk,
#                             "tokens": current_tokens
#                         }
#                     ))
#                     logger.debug("Created chunk %d for doc %s: header=%s",
#                                  chunk_index, doc.id, header)
#                     chunk_index += 1
#                     current_chunk = [sentence]
#                     current_tokens = sentence_tokens
#                 else:
#                     current_chunk.append(sentence)
#                     current_tokens += sentence_tokens
#         else:
#             # Line-based chunking with get_words
#             text_lines = metadata.get("texts", doc.text.splitlines())
#             current_chunk = []
#             current_tokens = 0
#             for line in text_lines:
#                 line_tokens = len(get_words(line))
#                 if current_tokens + line_tokens > max_tokens and current_chunk:
#                     chunk_text = "\n".join(current_chunk)
#                     chunked_docs.append(HeaderDocument(
#                         id=f"{doc.id}_chunk_{chunk_index}",
#                         metadata={
#                             "source_url": metadata.get("source_url", None),
#                             "header": header,
#                             "parent_header": parent_header,
#                             "header_level": metadata.get("header_level", 0) + 1,
#                             "content": chunk_text,
#                             "doc_index": doc_index,
#                             "chunk_index": chunk_index,
#                             "texts": current_chunk,
#                             "tokens": current_tokens
#                         }
#                     ))
#                     logger.debug("Created chunk %d for doc %s: header=%s",
#                                  chunk_index, doc.id, header)
#                     chunk_index += 1
#                     current_chunk = [line]
#                     current_tokens = line_tokens
#                 else:
#                     current_chunk.append(line)
#                     current_tokens += line_tokens

#         if current_chunk:
#             chunk_text = " ".join(
#                 current_chunk) if model else "\n".join(current_chunk)
#             chunked_docs.append(HeaderDocument(
#                 id=f"{doc.id}_chunk_{chunk_index}",
#                 text=chunk_text,
#                 metadata={
#                     "source_url": metadata.get("source_url", None),
#                     "header": header,
#                     "parent_header": parent_header,
#                     "header_level": metadata.get("header_level", 0) + 1,
#                     "content": chunk_text,
#                     "doc_index": doc_index,
#                     "chunk_index": chunk_index,
#                     "texts": current_chunk,
#                     "tokens": current_tokens
#                 }
#             ))
#             logger.debug("Created final chunk %d for doc %s: header=%s",
#                          chunk_index, doc.id, header)
#             chunk_index += 1

#     logger.info("Generated %d chunks from %d documents",
#                 len(chunked_docs), len(docs))
#     return chunked_docs


def split_sentences_with_separator_tuples(text: str, num_sentence: int = 1) -> List[Tuple[str, str]]:
    """Split text into sentences, preserving the separator after each sentence.

    Args:
        text: Input text to split.
        num_sentence: Number of sentences to combine into each chunk.

    Returns:
        List of tuples, each containing a sentence and its trailing separator.
    """
    if num_sentence < 1:
        raise ValueError("num_sentence must be a positive integer")

    sentences = sent_tokenize(text)
    if not sentences:
        return []

    adjusted_sentences = []
    i = 0
    current_pos = 0
    while i < len(sentences):
        current_sentence = sentences[i]
        # Find the sentence in the original text to determine its separator
        start_idx = text.find(current_sentence, current_pos)
        if start_idx == -1:
            # Fallback if sentence not found
            adjusted_sentences.append((current_sentence, " "))
            i += 1
            continue
        end_idx = start_idx + len(current_sentence)
        # Extract separator
        separator = ""
        if end_idx < len(text):
            next_sentence_idx = text.find(
                sentences[i + 1], end_idx) if i + 1 < len(sentences) else len(text)
            separator = text[end_idx:next_sentence_idx]
            if not separator.strip():  # If separator is only whitespace/newlines
                separator = separator if "\n" in separator else " "
            else:
                separator = " "  # Default to space for non-whitespace separators

        if is_list_marker(current_sentence) and i + 1 < len(sentences):
            combined = current_sentence + ' ' + sentences[i + 1]
            if is_list_sentence(combined):
                # Find the combined sentence in the original text
                combined_start = text.find(combined, current_pos)
                if combined_start != -1:
                    combined_end = combined_start + len(combined)
                    combined_separator = ""
                    if combined_end < len(text):
                        next_idx = text.find(
                            sentences[i + 2], combined_end) if i + 2 < len(sentences) else len(text)
                        combined_separator = text[combined_end:next_idx]
                        if not combined_separator.strip():
                            combined_separator = combined_separator if "\n" in combined_separator else " "
                        else:
                            combined_separator = " "
                    adjusted_sentences.append((combined, combined_separator))
                    current_pos = combined_end
                    i += 2
                    continue
        elif is_list_sentence(current_sentence):
            adjusted_sentences.append((current_sentence, separator))
        else:
            adjusted_sentences.append((current_sentence, separator))

        current_pos = end_idx
        i += 1

    # Combine sentences based on num_sentence
    combined_results = []
    for j in range(0, len(adjusted_sentences), num_sentence):
        chunk = adjusted_sentences[j:j + num_sentence]
        combined_sentence = ""
        for k, (sentence, sep) in enumerate(chunk):
            combined_sentence += sentence
            if k < len(chunk) - 1:  # Add separator except for the last sentence
                combined_sentence += sep
        # Use the last sentence's separator for the combined chunk
        final_separator = chunk[-1][1] if chunk else " "
        combined_results.append((combined_sentence, final_separator))

    return combined_results
