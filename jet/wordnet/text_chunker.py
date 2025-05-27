from typing import Union, List, Tuple
from nltk.tokenize import sent_tokenize
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
