from typing import Union, List, Tuple
from nltk.tokenize import sent_tokenize
from jet.logger import logger
from jet.vectors.document_types import HeaderDocument, HeaderMetadata
import re


def chunk_texts(texts: Union[str, List[str]], chunk_size: int = 128, chunk_overlap: int = 0) -> List[str]:
    """Chunk large texts into smaller segments with word overlap."""
    if isinstance(texts, str):
        texts = [texts]
    chunked_texts = []
    for text in texts:
        words = text.split()
        for i in range(0, len(words), chunk_size - chunk_overlap):
            start_idx = max(0, i)
            end_idx = min(len(words), i + chunk_size)
            chunked_texts.append(" ".join(words[start_idx:end_idx]))
            if end_idx == len(words):
                break  # Stop if we've reached the end of the text
    return chunked_texts


def chunk_sentences(texts: Union[str, List[str]], chunk_size: int = 5, sentence_overlap: int = 0) -> List[str]:
    """Chunk texts by sentences with sentence overlap."""
    if isinstance(texts, str):
        texts = [texts]
    chunked_texts = []
    # Sentence splitting regex: matches .!?, but avoids splitting at decimal points
    sentence_splitter = re.compile(r'(?<=[.!?])\s+(?=\w)')

    for text in texts:
        sentences = sentence_splitter.split(text.strip())
        sentences = [s.strip() for s in sentences if s.strip()
                     ]  # Remove empty sentences
        if len(sentences) > chunk_size:
            for i in range(0, len(sentences) - chunk_size + 1, chunk_size - sentence_overlap):
                start_idx = max(0, i)
                end_idx = min(len(sentences), i + chunk_size)
                chunked_texts.append(" ".join(sentences[start_idx:end_idx]))
        else:
            chunked_texts.append(text)
    return chunked_texts


def chunk_texts_with_indices(texts: Union[str, List[str]], chunk_size: int = 128, chunk_overlap: int = 0) -> Tuple[List[str], List[int]]:
    """Chunk large texts and track original document indices with word overlap."""
    if isinstance(texts, str):
        texts = [texts]
    chunked_texts = []
    doc_indices = []  # Tracks which document each chunk belongs to
    for doc_idx, text in enumerate(texts):
        words = text.split()
        for i in range(0, len(words), chunk_size - chunk_overlap):
            start_idx = max(0, i)
            end_idx = min(len(words), i + chunk_size)
            chunked_texts.append(" ".join(words[start_idx:end_idx]))
            doc_indices.append(doc_idx)
            if end_idx == len(words):
                break  # Stop if we've reached the end of the text
    return chunked_texts, doc_indices


def chunk_sentences_with_indices(texts: Union[str, List[str]], chunk_size: int = 5, sentence_overlap: int = 0) -> Tuple[List[str], List[int]]:
    """Chunk texts by sentences with sentence overlap and track original document indices."""
    if isinstance(texts, str):
        texts = [texts]
    chunked_texts = []
    doc_indices = []
    # Sentence splitting regex: matches .!?, but avoids splitting at decimal points
    sentence_splitter = re.compile(r'(?<=[.!?])\s+(?=\w)')

    for doc_idx, text in enumerate(texts):
        sentences = sentence_splitter.split(text.strip())
        sentences = [s.strip() for s in sentences if s.strip()
                     ]  # Remove empty sentences
        if len(sentences) > chunk_size:
            for i in range(0, len(sentences) - chunk_size + 1, chunk_size - sentence_overlap):
                start_idx = max(0, i)
                end_idx = min(len(sentences), i + chunk_size)
                chunked_texts.append(" ".join(sentences[start_idx:end_idx]))
                doc_indices.append(doc_idx)
        else:
            chunked_texts.append(text)
            doc_indices.append(doc_idx)
    return chunked_texts, doc_indices


def chunk_headers(docs: List[HeaderDocument], max_tokens: int = 500) -> List[HeaderDocument]:
    """
    Chunk documents into smaller HeaderDocument objects with generated headers.

    Args:
        docs: List of HeaderDocument objects to chunk.
        max_tokens: Maximum token count per chunk (approximated by word count).

    Returns:
        List of chunked HeaderDocument objects with headers and metadata.
    """
    logger.debug("Starting chunk_headers with %d documents", len(docs))
    chunked_docs: List[HeaderDocument] = []

    for doc in docs:
        chunk_index = 0  # Reset chunk_index for each document
        metadata = HeaderMetadata(**doc.metadata)
        text_lines = metadata.get("texts", doc.text.splitlines())
        current_chunk = []
        current_tokens = 0
        parent_header = metadata.get("parent_header", "")
        doc_index = metadata.get("doc_index", 0)

        for line in text_lines:
            # Approximate token count (1 word ≈ 1.3 tokens)
            line_tokens = len(line.split()) * 1.3
            if current_tokens + line_tokens > max_tokens and current_chunk:
                # Create a new chunk
                chunk_text = "\n".join(current_chunk)
                header = current_chunk[0][:100] + \
                    "..." if current_chunk else ""
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
                        "tokens": round(current_tokens, 2)
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

        # Add remaining chunk
        if current_chunk:
            chunk_text = "\n".join(current_chunk)
            header = current_chunk[0][:100] + "..." if current_chunk else ""
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
                    "tokens": round(current_tokens, 2)
                }
            ))
            logger.debug("Created final chunk %d for doc %s: header=%s",
                         chunk_index, doc.id, header)
            chunk_index += 1

    logger.info("Generated %d chunks from %d documents",
                len(chunked_docs), len(docs))
    return chunked_docs


def truncate_texts(
    texts: Union[str, List[str]],
    max_words: int
) -> Union[str, List[str]]:
    def truncate_single(text: str) -> str:
        sentences = sent_tokenize(text)
        truncated = []
        word_count = 0
        for sentence in sentences:
            words = sentence.split()
            if word_count + len(words) > max_words:
                break
            truncated.append(sentence)
            word_count += len(words)
        return " ".join(truncated)

    if isinstance(texts, str):
        return truncate_single(texts)
    return [truncate_single(t) for t in texts]
