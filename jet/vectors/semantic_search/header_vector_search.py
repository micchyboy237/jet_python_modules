import re
from typing import List, Optional, Union, Tuple, TypedDict, Iterator, Callable
import numpy as np
from sentence_transformers import SentenceTransformer
from jet.logger import logger
from jet.models.embeddings.base import generate_embeddings
from jet.models.model_registry.transformers.sentence_transformer_registry import SentenceTransformerRegistry
from jet.models.model_types import EmbedModelType
from jet.code.markdown_types.markdown_parsed_types import HeaderDoc, MarkdownToken
import logging
from jet.utils.text_constants import TEXT_CONTRACTIONS_EN

DEFAULT_EMBED_MODEL: EmbedModelType = 'all-MiniLM-L6-v2'
MAX_CONTENT_SIZE = 1000


class HeaderSearchMetadata(TypedDict):
    """Typed dictionary for search result metadata."""
    doc_index: int
    doc_id: str
    header: str
    level: Optional[int]
    parent_header: Optional[str]
    parent_level: Optional[int]
    start_idx: int
    end_idx: int
    chunk_idx: int
    header_similarity: float
    parent_similarity: float
    content_similarity: float
    num_tokens: int


class HeaderSearchResult(TypedDict):
    """Typed dictionary for search result structure."""
    rank: int
    score: float
    metadata: HeaderSearchMetadata
    content: str


def preprocess_text(
    text: str,
) -> str:
    if not text or not text.strip():
        return ""
    text = re.sub(r'\s+', ' ', text.strip())
    for contraction, expanded in TEXT_CONTRACTIONS_EN.items():
        text = re.sub(r'\b' + contraction + r'\b',
                      expanded, text, flags=re.IGNORECASE)
    text = text.lower()
    preserve_chars = {'-', '_'}
    pattern = r'[^a-z0-9\s' + ''.join(map(re.escape, preserve_chars)) + r']'
    text = re.sub(pattern, '', text)
    text = re.sub(r'\s+', ' ', text.strip())
    return text


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    return dot_product / (norm_a * norm_b) if norm_a * norm_b != 0 else 0.0


def collect_header_chunks(
    header_docs: List[HeaderDoc],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    tokenizer: Optional[Callable[[str], int]] = None
) -> Tuple[List[int], List[str], List[str], List[Tuple[int, str, str, str, int, int, int]]]:
    """
    Collect chunked contents for each header along with metadata, preserving original texts.
    Args:
        header_docs: List of HeaderDoc objects to process
        chunk_size: Size of content chunks
        chunk_overlap: Overlap between chunks
        tokenizer: Optional callable to count tokens in text. Defaults to regex-based word and punctuation counting.
    Returns:
        Tuple of (doc_indices, headers, parent_headers, contents_with_indices)
        where contents_with_indices = List of (doc_index, header, content_chunk, original_content_chunk, start_idx, end_idx, num_tokens)
    """
    def default_tokenizer(text): return len(
        re.findall(r'\b\w+\b|[^\w\s]', text))
    tokenizer = tokenizer or default_tokenizer
    doc_indices, headers, parent_headers = [], [], []
    contents_with_indices = []
    for header_doc in header_docs:
        doc_index = header_doc['doc_index']
        original_header = header_doc['header']
        header = preprocess_text(original_header)
        original_parent = header_doc.get('parent_header', '')
        parent_header = preprocess_text(
            original_parent) if original_parent else ""
        original_content = header_doc['content']
        content = preprocess_text(original_content)
        doc_indices.append(doc_index)
        headers.append(header)
        parent_headers.append(parent_header)
        for start in range(0, len(original_content), chunk_size - chunk_overlap):
            original_chunk = original_content[start:start + chunk_size]
            chunk = preprocess_text(original_chunk)
            if chunk.strip():
                end = start + len(original_chunk)
                num_tokens = tokenizer(original_chunk)
                contents_with_indices.append(
                    (doc_index, header, chunk, original_chunk, start, end, num_tokens))
    return doc_indices, headers, parent_headers, contents_with_indices


def compute_weighted_similarity(
    query_vector: np.ndarray,
    header_vector: np.ndarray,
    parent_vector: np.ndarray,
    content_vector: Optional[np.ndarray]
) -> Tuple[float, float, float, float]:
    """
    Compute weighted similarity score and individual scores for a header based on its components.
    Args:
        query_vector: Encoded query vector
        header_vector: Encoded header vector
        parent_vector: Encoded parent header vector
        content_vector: Encoded content vector (if available)
    Returns:
        Tuple of (weighted_similarity, header_similarity, parent_similarity, content_similarity)
    """
    header_sim = cosine_similarity(query_vector, header_vector)
    parent_sim = cosine_similarity(
        query_vector, parent_vector) if np.any(parent_vector) else 0.0
    content_sim = 0.0
    if content_vector is not None:
        content_sim = cosine_similarity(query_vector, content_vector)
    weighted_sim = 0.4 * header_sim + 0.2 * parent_sim + 0.4 * content_sim
    return weighted_sim, header_sim, parent_sim, content_sim


def merge_results(
    results: List[HeaderSearchResult],
    tokenizer: Optional[Callable[[str], int]] = None
) -> List[HeaderSearchResult]:
    """
    Merge adjacent chunks from the same header into a single result, preserving order and metadata.
    Args:
        results: List of HeaderSearchResult dictionaries, potentially containing adjacent chunks.
        tokenizer: Optional callable to count tokens in text. Defaults to regex-based word and punctuation counting.
    Returns:
        List of HeaderSearchResult dictionaries with adjacent chunks merged.
    """
    if not results:
        return []
    def default_tokenizer(text): return len(
        re.findall(r'\b\w+\b|[^\w\s]', text))
    tokenizer = tokenizer or default_tokenizer
    grouped: dict[int, List[HeaderSearchResult]] = {}
    for result in results:
        doc_index = result["metadata"]["doc_index"]
        if doc_index not in grouped:
            grouped[doc_index] = []
        grouped[doc_index].append(result)
    merged_results: List[HeaderSearchResult] = []
    for doc_index, chunks in grouped.items():
        chunks.sort(key=lambda x: x["metadata"]["start_idx"])
        current_chunk = chunks[0]
        merged_content = current_chunk["content"]
        start_idx = current_chunk["metadata"]["start_idx"]
        end_idx = current_chunk["metadata"]["end_idx"]
        total_score = current_chunk["score"]
        header_sim = current_chunk["metadata"]["header_similarity"]
        parent_sim = current_chunk["metadata"]["parent_similarity"]
        content_sims = [current_chunk["metadata"]["content_similarity"]]
        chunk_count = 1
        tokens = tokenizer(merged_content)
        for next_chunk in chunks[1:]:
            next_start = next_chunk["metadata"]["start_idx"]
            next_end = next_chunk["metadata"]["end_idx"]
            next_content = next_chunk["content"]
            if next_start <= end_idx:
                new_end = max(end_idx, next_end)
                overlap = end_idx - next_start
                additional_content = next_content[overlap:
                                                  ] if overlap > 0 else next_content
                merged_content += additional_content
                end_idx = new_end
                total_score += next_chunk["score"]
                content_sims.append(
                    next_chunk["metadata"]["content_similarity"])
                chunk_count += 1
                tokens = tokenizer(merged_content)
            else:
                avg_score = total_score / chunk_count
                avg_content_sim = sum(content_sims) / chunk_count
                merged_results.append({
                    "rank": current_chunk["rank"],
                    "score": avg_score,
                    "metadata": {
                        "doc_index": doc_index,
                        "doc_id": current_chunk["metadata"]["doc_id"],
                        "header": current_chunk["metadata"]["header"],
                        "level": current_chunk["metadata"]["level"],
                        "parent_header": current_chunk["metadata"]["parent_header"],
                        "parent_level": current_chunk["metadata"]["parent_level"],
                        "start_idx": start_idx,
                        "end_idx": end_idx,
                        "chunk_idx": 0,
                        "header_similarity": header_sim,
                        "parent_similarity": parent_sim,
                        "content_similarity": avg_content_sim,
                        "num_tokens": tokens
                    },
                    "content": merged_content,
                })
                current_chunk = next_chunk
                merged_content = current_chunk["content"]
                start_idx = current_chunk["metadata"]["start_idx"]
                end_idx = current_chunk["metadata"]["end_idx"]
                total_score = current_chunk["score"]
                header_sim = current_chunk["metadata"]["header_similarity"]
                parent_sim = current_chunk["metadata"]["parent_similarity"]
                content_sims = [current_chunk["metadata"]
                                ["content_similarity"]]
                chunk_count = 1
                tokens = tokenizer(merged_content)
        avg_score = total_score / chunk_count
        avg_content_sim = sum(content_sims) / chunk_count
        merged_results.append({
            "rank": current_chunk["rank"],
            "score": avg_score,
            "metadata": {
                "doc_index": doc_index,
                "doc_id": current_chunk["metadata"]["doc_id"],
                "header": current_chunk["metadata"]["header"],
                "level": current_chunk["metadata"]["level"],
                "parent_header": current_chunk["metadata"]["parent_header"],
                "parent_level": current_chunk["metadata"]["parent_level"],
                "start_idx": start_idx,
                "end_idx": end_idx,
                "chunk_idx": 0,
                "header_similarity": header_sim,
                "parent_similarity": parent_sim,
                "content_similarity": avg_content_sim,
                "num_tokens": tokens
            },
            "content": merged_content,
        })
    merged_results.sort(key=lambda x: x["score"], reverse=True)
    for i, result in enumerate(merged_results, 1):
        result["rank"] = i
    return merged_results


def search_headers(
    header_docs: List[HeaderDoc],
    query: str,
    top_k: int = 5,
    embed_model: EmbedModelType = DEFAULT_EMBED_MODEL,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    threshold: float = 0.0,
    tokenizer: Optional[Callable[[str], int]] = None,
    split_chunks: bool = False
) -> Iterator[HeaderSearchResult]:
    """
    Search headers using vector similarity on chunked contents + header metadata.
    Yields up to top_k results iteratively that meet the threshold, preserving original texts.
    Args:
        header_docs: List of HeaderDoc objects to search
        query: Search query string
        top_k: Maximum number of results to yield
        embed_model: Embedding model to use
        chunk_size: Size of content chunks
        chunk_overlap: Overlap between chunks
        threshold: Minimum similarity score for results
        tokenizer: Optional callable to count tokens in text. Defaults to regex-based word and punctuation counting.
        split_chunks: If True, return individual chunks; if False, merge adjacent chunks from the same header.
    Returns:
        Iterator of HeaderSearchResult dictionaries (ranked by similarity)
    """
    def default_tokenizer(text): return len(
        re.findall(r'\b\w+\b|[^\w\s]', text))
    tokenizer = tokenizer or default_tokenizer
    query_processed = preprocess_text(query)
    doc_indices, headers, parent_headers, chunk_data = collect_header_chunks(
        header_docs, chunk_size, chunk_overlap, tokenizer)
    if not chunk_data:
        return
    unique_docs = list(dict.fromkeys(doc_indices))
    header_texts = [headers[doc_indices.index(idx)] for idx in unique_docs]
    parent_texts = [
        parent_headers[doc_indices.index(idx)] for idx in unique_docs]
    chunk_texts = [chunk for _, _, chunk, _, _, _, _ in chunk_data]
    all_texts = [query_processed] + header_texts + parent_texts + chunk_texts
    logger.info(
        f"Generating embeddings for {len(all_texts)} texts:\n"
        f"  1 query\n"
        f"  {len(header_texts)} headers\n"
        f"  {len(parent_texts)} parent headers\n"
        f"  {len(chunk_texts)} chunks"
    )
    all_vectors = generate_embeddings(
        all_texts,
        embed_model,
        return_format="numpy",
        batch_size=32,
        show_progress=True
    )
    query_vector = all_vectors[0]
    num_headers = len(header_texts)
    num_parents = len(parent_texts)
    header_vectors = all_vectors[1:num_headers + 1]
    parent_vectors = all_vectors[num_headers + 1:num_headers + 1 + num_parents]
    content_vectors = all_vectors[num_headers + 1 + num_parents:]
    results: List[HeaderSearchResult] = []
    chunk_counts = {}
    for i, (doc_index, header, chunk, original_chunk, start_idx, end_idx, num_tokens) in enumerate(chunk_data):
        unique_doc_idx = unique_docs.index(doc_index)
        header_doc = next(
            hd for hd in header_docs if hd['doc_index'] == doc_index)
        content_vector = content_vectors[i]
        weighted_sim, header_sim, parent_sim, content_sim = compute_weighted_similarity(
            query_vector, header_vectors[unique_doc_idx], parent_vectors[unique_doc_idx], content_vector
        )
        if weighted_sim >= threshold:
            chunk_counts[doc_index] = chunk_counts.get(doc_index, -1) + 1
            result = {
                "rank": 0,
                "score": float(weighted_sim),
                "metadata": {
                    "doc_index": doc_index,
                    "doc_id": header_doc['doc_id'],
                    "header": header_doc['header'],
                    "level": header_doc['level'],
                    "parent_header": header_doc['parent_header'],
                    "parent_level": header_doc['parent_level'],
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "chunk_idx": chunk_counts[doc_index],
                    "header_similarity": float(header_sim),
                    "parent_similarity": float(parent_sim),
                    "content_similarity": float(content_sim),
                    "num_tokens": num_tokens
                },
                "content": original_chunk,
            }
            results.append(result)
    results.sort(key=lambda x: x["score"], reverse=True)
    if not split_chunks:
        results = merge_results(results, tokenizer)
    for i, result in enumerate(results[:top_k], 1):
        result["rank"] = i
        yield result
