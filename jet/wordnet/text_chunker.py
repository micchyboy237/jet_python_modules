import uuid
import re
from nltk.tokenize import sent_tokenize
from typing import TypedDict, Union, List, Tuple, Optional
from jet.llm.models import OLLAMA_MODEL_NAMES
from jet.logger import logger
# from jet.vectors.document_types import HeaderDocument, HeaderMetadata
from jet._token.token_utils import get_last_n_tokens_and_decode, get_model_max_tokens, get_tokenizer, get_tokenizer_fn
from jet.wordnet.sentence import split_sentences, is_list_marker, is_list_sentence
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


class ChunkResult(TypedDict):
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


def chunk_texts(
    texts: Union[str, List[str]],
    chunk_size: int = 128,
    chunk_overlap: int = 0,
    model: Optional[OLLAMA_MODEL_NAMES] = None,
    buffer: int = 0,
    strict_sentences: bool = False,
    min_chunk_size: int = 32,  # New parameter for minimum chunk size
) -> List[str]:
    if min_chunk_size > chunk_size:
        min_chunk_size = chunk_size
    if isinstance(texts, str):
        texts = [texts]
    chunked_texts = []
    for text in texts:
        sentences = split_sentences(text)
        if not sentences:
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
        size_fn = get_tokenizer_fn(model) if model else get_words
        tokenizer = get_tokenizer(model) if model else None
        effective_chunk_size = chunk_size - buffer
        if not strict_sentences and model:
            # Token-based chunking without sentence boundaries
            tokens = size_fn(text)
            if not tokens:
                continue
            step = max(1, chunk_size - chunk_overlap - buffer)
            for i in range(0, len(tokens), step):
                chunk_tokens = tokens[i:i + chunk_size - buffer]
                if chunk_tokens:
                    chunk = tokenizer.decode(chunk_tokens).strip() if model else " ".join(chunk_tokens).strip()
                    chunk_size_tokens = len(size_fn(chunk))
                    # Skip chunks smaller than min_chunk_size if not the last chunk
                    if chunk_size_tokens < min_chunk_size and i + chunk_size - buffer < len(tokens):
                        continue
                    chunked_texts.append(chunk)
            # Merge last chunk if too small
            if len(chunked_texts) > 1 and len(size_fn(chunked_texts[-1])) < min_chunk_size:
                last_chunk = chunked_texts.pop()
                prev_chunk = chunked_texts[-1]
                prev_chunk_last_n_tokens_string = get_last_n_tokens_and_decode(prev_chunk, tokenizer, len(size_fn(last_chunk)))
                is_covered_by_prev_chunk = last_chunk == prev_chunk_last_n_tokens_string
                if not is_covered_by_prev_chunk:
                    chunked_texts[-1] = prev_chunk + " " + last_chunk
            continue
        current_chunk = []
        current_separators = []
        current_size = 0
        for sentence, separator in sentence_pairs:
            sentence_size = len(size_fn(sentence))
            if sentence_size > effective_chunk_size:
                # Split large sentence
                sub_sentences = split_large_sentence(sentence, effective_chunk_size, size_fn)
                for sub_sentence in sub_sentences:
                    sub_size = len(size_fn(sub_sentence))
                    if current_size + sub_size > effective_chunk_size and current_chunk:
                        chunk_content = build_chunk(current_chunk, current_separators)
                        final_size = len(size_fn(chunk_content))
                        if final_size >= min_chunk_size or not chunked_texts:
                            chunked_texts.append(chunk_content)
                        overlap_sentences, overlap_separators, overlap_size = get_overlap_sentences(
                            current_chunk, current_separators, chunk_overlap, size_fn
                        )
                        current_chunk = overlap_sentences
                        current_separators = overlap_separators
                        current_size = overlap_size
                    current_chunk.append(sub_sentence)
                    current_separators.append(" ")
                    current_size += sub_size
            else:
                if current_size + sentence_size > effective_chunk_size and current_chunk:
                    chunk_content = build_chunk(current_chunk, current_separators)
                    final_size = len(size_fn(chunk_content))
                    if final_size >= min_chunk_size or not chunked_texts:
                        chunked_texts.append(chunk_content)
                    overlap_sentences, overlap_separators, overlap_size = get_overlap_sentences(
                        current_chunk, current_separators, chunk_overlap, size_fn
                    )
                    current_chunk = overlap_sentences
                    current_separators = overlap_separators
                    current_size = overlap_size
                current_chunk.append(sentence)
                current_separators.append(separator)
                current_size += sentence_size
        if current_chunk:
            chunk_content = build_chunk(current_chunk, current_separators)
            final_size = len(size_fn(chunk_content))
            # Merge with previous chunk if too small and not the only chunk
            if final_size < min_chunk_size and len(chunked_texts) > 0:
                prev_chunk = chunked_texts[-1]
                prev_chunk_last_n_tokens_string = get_last_n_tokens_and_decode(prev_chunk, tokenizer, final_size)
                is_covered_by_prev_chunk = chunk_content == prev_chunk_last_n_tokens_string
                if not is_covered_by_prev_chunk:
                    chunked_texts[-1] = prev_chunk + " " + chunk_content
            else:
                chunked_texts.append(chunk_content)
    return chunked_texts


def chunk_texts_with_data(
    texts: Union[str, List[str]],
    chunk_size: int = 128,
    chunk_overlap: int = 0,
    model: Optional[OLLAMA_MODEL_NAMES] = None,
    doc_ids: Optional[List[str]] = None,
    buffer: int = 0,
    strict_sentences: bool = False,
    min_chunk_size: int = 32,  # New parameter for minimum chunk size
) -> List[ChunkResult]:
    if min_chunk_size > chunk_size:
        min_chunk_size = chunk_size
    if isinstance(texts, str):
        texts = [texts]
        doc_indices = [0] * len(texts)
    else:
        doc_indices = list(range(len(texts)))
    chunks: List[ChunkResult] = []
    effective_chunk_size = chunk_size - buffer
    for i, (doc_index, text) in enumerate(zip(doc_indices, texts)):
        sentences = split_sentences(text)
        if not sentences:
            continue
        size_fn = get_tokenizer_fn(model) if model else get_words
        tokenizer = get_tokenizer(model) if model else None
        if not strict_sentences and model:
            tokens = size_fn(text)
            if not tokens:
                continue
            chunk_index = 0
            doc_id = doc_ids[i] if doc_ids and i < len(doc_ids) else str(uuid.uuid4())
            step = max(1, chunk_size - chunk_overlap - buffer)
            for j in range(0, len(tokens), step):
                chunk_tokens = tokens[j:j + chunk_size - buffer]
                if chunk_tokens:
                    chunk_content = tokenizer.decode(chunk_tokens).strip()
                    chunk_size_tokens = len(size_fn(chunk_content))
                    # Skip chunks smaller than min_chunk_size if not the last chunk
                    if chunk_size_tokens < min_chunk_size and j + chunk_size - buffer < len(tokens):
                        continue
                    overlap_start_idx = None
                    overlap_end_idx = None
                    if chunk_overlap > 0 and j + chunk_size - buffer < len(tokens):
                        overlap_tokens = tokens[j + chunk_size - chunk_overlap - buffer:j + chunk_size - buffer]
                        if overlap_tokens:
                            overlap_content = tokenizer.decode(overlap_tokens).strip()
                            overlap_start_idx = j + chunk_size - chunk_overlap - buffer
                            overlap_end_idx = j + chunk_size - buffer
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
            # Merge last chunk if too small
            if len(chunks) > 1 and chunks[-1]["num_tokens"] < min_chunk_size:
                last_chunk = chunks.pop()
                prev_chunk = chunks[-1]
                prev_chunk_last_n_tokens_string = get_last_n_tokens_and_decode(prev_chunk["content"], tokenizer, last_chunk["num_tokens"])
                is_covered_by_prev_chunk = last_chunk["content"] == prev_chunk_last_n_tokens_string
                if not is_covered_by_prev_chunk:
                    prev_chunk["content"] += " " + last_chunk["content"]
                    prev_chunk["num_tokens"] = len(size_fn(prev_chunk["content"]))
                    prev_chunk["end_idx"] = last_chunk["end_idx"]
            continue
        processed_sentences = []
        for sentence in sentences:
            processed_sentences.extend(split_large_sentence(
                sentence, effective_chunk_size, size_fn))
        sentence_pairs = []
        current_pos = 0
        line_idx = 0
        idx = 0
        while idx < len(processed_sentences):
            current_sentence = processed_sentences[idx]
            start_idx = text.find(current_sentence, current_pos)
            if start_idx == -1:
                sentence_pairs.append(
                    (current_sentence, " ", start_idx, current_pos, line_idx))
                idx += 1
                continue
            end_idx = start_idx + len(current_sentence)
            separator = text[end_idx:text.find(
                processed_sentences[idx + 1], end_idx) if idx + 1 < len(processed_sentences) else len(text)]
            separator = normalize_separator(separator)
            if is_list_marker(current_sentence) and idx + 1 < len(processed_sentences):
                combined = current_sentence + " " + processed_sentences[idx + 1]
                if is_list_sentence(combined):
                    combined_start = text.find(combined, current_pos)
                    if combined_start != -1:
                        combined_end = combined_start + len(combined)
                        combined_separator = text[combined_end:text.find(
                            processed_sentences[idx + 2], combined_end) if idx + 2 < len(processed_sentences) else len(text)]
                        combined_separator = normalize_separator(combined_separator)
                        sentence_pairs.append(
                            (combined, combined_separator, combined_start, combined_end, line_idx))
                        current_pos = combined_end + len(combined_separator)
                        line_idx += combined.count('\n')
                        idx += 2
                        continue
            sentence_pairs.append(
                (current_sentence, separator, start_idx, end_idx, line_idx))
            line_idx += current_sentence.count('\n') + separator.count('\n')
            current_pos = end_idx + len(separator)
            idx += 1
        current_chunk, current_separators = [], []
        current_size = 0
        chunk_index = 0
        chunk_start_idx = 0
        chunk_line_idx = 0
        doc_id = doc_ids[i] if doc_ids and i < len(doc_ids) else str(uuid.uuid4())
        for sentence, separator, start_idx, end_idx, line_idx in sentence_pairs:
            sentence_size = len(size_fn(sentence))
            if sentence_size > effective_chunk_size:
                logger.warning(
                    f"Skipping sentence with {sentence_size} tokens exceeding effective_chunk_size {effective_chunk_size}")
                continue
            if current_size + sentence_size > effective_chunk_size and current_chunk:
                chunk_content = build_chunk(current_chunk, current_separators)
                final_size = len(size_fn(chunk_content))
                if final_size >= min_chunk_size or len(chunks) == 0:
                    chunks.append({
                        "id": str(uuid.uuid4()),
                        "doc_id": doc_id,
                        "doc_index": doc_index,
                        "chunk_index": chunk_index,
                        "num_tokens": final_size,
                        "content": chunk_content,
                        "start_idx": chunk_start_idx,
                        "end_idx": chunk_start_idx + len(chunk_content),
                        "line_idx": chunk_line_idx,
                        "overlap_start_idx": None,
                        "overlap_end_idx": None
                    })
                    chunk_index += 1
                overlap_sentences, overlap_separators, overlap_size = get_overlap_sentences(
                    current_chunk, current_separators, chunk_overlap, size_fn
                )
                overlap_start_idx = None
                if overlap_sentences:
                    first_overlap_idx = next((i for i, s in enumerate(
                        current_chunk) if s == overlap_sentences[0]), None)
                    if first_overlap_idx is not None:
                        overlap_start_idx = sentence_pairs[first_overlap_idx][2]
                chunk_start_idx = overlap_start_idx if overlap_start_idx is not None else start_idx
                chunk_line_idx = line_idx
                current_chunk = overlap_sentences
                current_separators = overlap_separators
                current_size = overlap_size
            current_chunk.append(sentence)
            current_separators.append(separator)
            current_size += sentence_size
            if not current_chunk:
                chunk_start_idx = start_idx
                chunk_line_idx = line_idx
        if current_chunk:
            chunk_content = build_chunk(current_chunk, current_separators)
            final_size = len(size_fn(chunk_content))
            # Merge with previous chunk if too small and not the only chunk
            if final_size < min_chunk_size and len(chunks) > 0:
                prev_chunk = chunks[-1]
                prev_chunk["content"] += " " + chunk_content
                prev_chunk["num_tokens"] = len(size_fn(prev_chunk["content"]))
                prev_chunk["end_idx"] = chunk_start_idx + len(chunk_content)
            else:
                chunks.append({
                    "id": str(uuid.uuid4()),
                    "doc_id": doc_id,
                    "doc_index": doc_index,
                    "chunk_index": chunk_index,
                    "num_tokens": final_size,
                    "content": chunk_content,
                    "start_idx": chunk_start_idx,
                    "end_idx": chunk_start_idx + len(chunk_content),
                    "line_idx": chunk_line_idx,
                    "overlap_start_idx": None,
                    "overlap_end_idx": None
                })
    return chunks


def chunk_texts_sliding_window(
    texts: Union[str, List[str]],
    chunk_size: int = 128,
    step_size: int = 96,
    model: Optional[OLLAMA_MODEL_NAMES] = None,
    doc_ids: Optional[List[str]] = None,
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
        doc_ids: Optional list of document IDs.
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
        doc_id = doc_ids[i] if doc_ids and i < len(doc_ids) else str(uuid.uuid4())
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


def truncate_texts(
    texts: str | List[str],
    model: OLLAMA_MODEL_NAMES,
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


def chunk_sentences(texts: Union[str, List[str]], chunk_size: int = 5, chunk_overlap: int = 0, model: Optional[OLLAMA_MODEL_NAMES] = None) -> List[str]:
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


def chunk_sentences_with_indices(texts: Union[str, List[str]], chunk_size: int = 5, chunk_overlap: int = 0, model: Optional[OLLAMA_MODEL_NAMES] = None) -> Tuple[List[str], List[int]]:
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


# def chunk_headers(docs: List[HeaderDocument], max_tokens: int = 500, model: Optional[OLLAMA_MODEL_NAMES] = None) -> List[HeaderDocument]:
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
