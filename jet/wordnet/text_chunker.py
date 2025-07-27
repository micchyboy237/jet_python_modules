from jet.models.tokenizer.base import get_tokenizer_fn
from typing import Union, List, Tuple, Optional
from nltk.tokenize import sent_tokenize
from jet.logger import logger
from jet.models.utils import get_context_size
from jet.vectors.document_types import HeaderDocument, HeaderMetadata
from jet.models.tokenizer.base import detokenize, get_tokenizer_fn, get_tokenizer
from jet.models.model_types import ModelType
import re
import numpy as np

from jet.wordnet.sentence import split_sentences, split_sentences_with_separators
from jet.wordnet.words import get_words


def chunk_texts(texts: Union[str, List[str]], chunk_size: int = 128, chunk_overlap: int = 0, model: Optional[ModelType] = None) -> List[str]:
    """Chunk large texts into smaller segments with word or token overlap based on model presence.

    Args:
        texts: Single string or list of strings to chunk.
        chunk_size: Number of words or tokens per chunk.
        chunk_overlap: Number of words or tokens to overlap between chunks.
        model: Optional LLM model name for token-based chunking.
    """
    if isinstance(texts, str):
        texts = [texts]

    chunked_texts = []
    if model:
        # Token-based chunking
        tokenize_fn = get_tokenizer_fn(model)
        for text in texts:
            token_ids = tokenize_fn(text)
            for i in range(0, len(token_ids), chunk_size - chunk_overlap):
                start_idx = max(0, i)
                end_idx = min(len(token_ids), i + chunk_size)
                chunk_tokens = token_ids[start_idx:end_idx]
                chunked_texts.append(detokenize(chunk_tokens, model))
                if end_idx == len(token_ids):
                    break
    else:
        # Word-based chunking
        for text in texts:
            words = get_words(text)
            for i in range(0, len(words), chunk_size - chunk_overlap):
                start_idx = max(0, i)
                end_idx = min(len(words), i + chunk_size)
                chunked_texts.append(" ".join(words[start_idx:end_idx]))
                if end_idx == len(words):
                    break
    return chunked_texts


def chunk_sentences(texts: Union[str, List[str]], chunk_size: int = 5, sentence_overlap: int = 0, model: Optional[ModelType] = None) -> List[str]:
    """Chunk texts by sentences with sentence overlap, using tokens if model is provided, preserving original separators.

    Args:
        texts: Single string or list of strings to chunk.
        chunk_size: Number of sentences (non-model) or tokens (model) per chunk.
        sentence_overlap: Number of sentences to overlap.
        model: Optional LLM model name for token-based chunking.
    """
    if isinstance(texts, str):
        texts = [texts]
    chunked_texts = []

    if model:
        tokenize_fn = get_tokenizer_fn(model)
        for text in texts:
            sentence_pairs = split_sentences_with_separators(text)
            sentences = [s for s, _ in sentence_pairs]
            separators = [sep for _, sep in sentence_pairs]
            if not sentences:
                logger.debug(f"No sentences found for text: {text}")
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
                    overlap_start = max(0, i - sentence_overlap)
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
            sentence_pairs = split_sentences_with_separators(text)
            sentences = [s for s, _ in sentence_pairs]
            separators = [sep for _, sep in sentence_pairs]
            sentences = [s.strip() for s in sentences if s.strip()]
            if not sentences:
                continue
            if len(sentences) > chunk_size:
                for i in range(0, len(sentences) - chunk_size + 1, chunk_size - sentence_overlap):
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


def chunk_texts_with_indices(texts: Union[str, List[str]], chunk_size: int = 128, chunk_overlap: int = 0, model: Optional[ModelType] = None) -> Tuple[List[str], List[int]]:
    """Chunk large texts and track original document indices with word or token overlap.

    Args:
        texts: Single string or list of strings to chunk.
        chunk_size: Number of words or tokens per chunk.
        chunk_overlap: Number of words or tokens to overlap.
        model: Optional LLM model name for token-based chunking.
    """
    if isinstance(texts, str):
        texts = [texts]

    chunked_texts = []
    doc_indices = []
    if model:
        tokenize_fn = get_tokenizer_fn(model)
        for doc_idx, text in enumerate(texts):
            token_ids = tokenize_fn(text)
            for i in range(0, len(token_ids), chunk_size - chunk_overlap):
                start_idx = max(0, i)
                end_idx = min(len(token_ids), i + chunk_size)
                chunk_tokens = token_ids[start_idx:end_idx]
                chunked_texts.append(detokenize(chunk_tokens, model))
                doc_indices.append(doc_idx)
                if end_idx == len(token_ids):
                    break
    else:
        for doc_idx, text in enumerate(texts):
            words = get_words(text)
            for i in range(0, len(words), chunk_size - chunk_overlap):
                start_idx = max(0, i)
                end_idx = min(len(words), i + chunk_size)
                chunked_texts.append(" ".join(words[start_idx:end_idx]))
                doc_indices.append(doc_idx)
                if end_idx == len(words):
                    break
    return chunked_texts, doc_indices


def chunk_sentences_with_indices(texts: Union[str, List[str]], chunk_size: int = 5, sentence_overlap: int = 0, model: Optional[ModelType] = None) -> Tuple[List[str], List[int]]:
    """Chunk texts by sentences with sentence overlap and track original document indices, using tokens if model is provided.

    Args:
        texts: Single string or list of strings to chunk.
        chunk_size: Number of sentences or tokens per chunk.
        sentence_overlap: Number of sentences or tokens to overlap.
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
            sentences = sent_tokenize(text.strip())
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
                    overlap_start = max(0, i - sentence_overlap)
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
                for i in range(0, len(sentences) - chunk_size + 1, chunk_size - sentence_overlap):
                    start_idx = max(0, i)
                    end_idx = min(len(sentences), i + chunk_size)
                    chunked_texts.append(
                        " ".join(sentences[start_idx:end_idx]))
                    doc_indices.append(doc_idx)
            else:
                chunked_texts.append(text)
                doc_indices.append(doc_idx)
    return chunked_texts, doc_indices


def chunk_headers(docs: List[HeaderDocument], max_tokens: int = 500, model: Optional[ModelType] = None) -> List[HeaderDocument]:
    """Chunk HeaderDocument list into smaller segments based on token count or lines, ensuring complete sentences when model is provided.

    Args:
        docs: List of HeaderDocument objects to chunk.
        max_tokens: Maximum number of tokens or lines per chunk.
        model: Optional LLM model name for token-based chunking.
    """
    logger.debug("Starting chunk_headers with %d documents", len(docs))
    chunked_docs: List[HeaderDocument] = []

    for doc in docs:
        chunk_index = 0
        metadata = HeaderMetadata(**doc.metadata)
        parent_header = metadata.get("parent_header", "")
        doc_index = metadata.get("doc_index", 0)
        # Use original header from metadata
        header = metadata.get("header", "")

        if model:
            # Token-based chunking with sentence boundaries
            tokenize_fn = get_tokenizer_fn(model)
            sentences = sent_tokenize(doc.text)
            current_chunk = []
            current_tokens = 0
            for sentence in sentences:
                sentence_tokens = len(tokenize_fn(sentence))
                if current_tokens + sentence_tokens > max_tokens and current_chunk:
                    chunk_text = " ".join(current_chunk)
                    chunked_docs.append(HeaderDocument(
                        id=f"{doc.id}_chunk_{chunk_index}",
                        text=chunk_text,
                        metadata={
                            "source_url": metadata.get("source_url", None),
                            "header": header,
                            "parent_header": parent_header,
                            "header_level": metadata.get("header_level", 0) + 1,
                            "content": chunk_text,
                            "doc_index": doc_index,
                            "chunk_index": chunk_index,
                            "texts": current_chunk,
                            "tokens": current_tokens
                        }
                    ))
                    logger.debug("Created chunk %d for doc %s: header=%s",
                                 chunk_index, doc.id, header)
                    chunk_index += 1
                    current_chunk = [sentence]
                    current_tokens = sentence_tokens
                else:
                    current_chunk.append(sentence)
                    current_tokens += sentence_tokens
        else:
            # Line-based chunking with get_words
            text_lines = metadata.get("texts", doc.text.splitlines())
            current_chunk = []
            current_tokens = 0
            for line in text_lines:
                line_tokens = len(get_words(line))
                if current_tokens + line_tokens > max_tokens and current_chunk:
                    chunk_text = "\n".join(current_chunk)
                    chunked_docs.append(HeaderDocument(
                        id=f"{doc.id}_chunk_{chunk_index}",
                        metadata={
                            "source_url": metadata.get("source_url", None),
                            "header": header,
                            "parent_header": parent_header,
                            "header_level": metadata.get("header_level", 0) + 1,
                            "content": chunk_text,
                            "doc_index": doc_index,
                            "chunk_index": chunk_index,
                            "texts": current_chunk,
                            "tokens": current_tokens
                        }
                    ))
                    logger.debug("Created chunk %d for doc %s: header=%s",
                                 chunk_index, doc.id, header)
                    chunk_index += 1
                    current_chunk = [line]
                    current_tokens = line_tokens
                else:
                    current_chunk.append(line)
                    current_tokens += line_tokens

        if current_chunk:
            chunk_text = " ".join(
                current_chunk) if model else "\n".join(current_chunk)
            chunked_docs.append(HeaderDocument(
                id=f"{doc.id}_chunk_{chunk_index}",
                text=chunk_text,
                metadata={
                    "source_url": metadata.get("source_url", None),
                    "header": header,
                    "parent_header": parent_header,
                    "header_level": metadata.get("header_level", 0) + 1,
                    "content": chunk_text,
                    "doc_index": doc_index,
                    "chunk_index": chunk_index,
                    "texts": current_chunk,
                    "tokens": current_tokens
                }
            ))
            logger.debug("Created final chunk %d for doc %s: header=%s",
                         chunk_index, doc.id, header)
            chunk_index += 1

    logger.info("Generated %d chunks from %d documents",
                len(chunked_docs), len(docs))
    return chunked_docs


def truncate_texts(texts: str | list[str], model: ModelType, max_tokens: Optional[int] = None) -> list[str]:
    """
    Truncates texts that exceed the max_tokens limit.

    Args:
        texts (str | list[str]): A list of texts to be truncated.
        model (str): The model name for tokenization.
        max_tokens (int): The maximum number of tokens allowed per text.

    Returns:
        list[str]: A list of truncated texts.
    """
    tokenizer = get_tokenizer(model)

    if not max_tokens:
        max_tokens = get_context_size(model)

    if isinstance(texts, str):
        texts = [texts]

    tokenized_texts = tokenizer.batch_encode_plus(texts, return_tensors=None)
    tokenized_texts = tokenized_texts["input_ids"]
    truncated_texts = []

    for text, tokens in zip(texts, tokenized_texts):
        if len(tokens) > max_tokens:
            truncated_text = tokenizer.decode(
                tokens[:max_tokens], skip_special_tokens=True)
            truncated_texts.append(truncated_text)
        else:
            truncated_texts.append(text)

    return truncated_texts
