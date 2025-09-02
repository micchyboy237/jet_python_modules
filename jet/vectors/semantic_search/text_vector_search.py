import re
from typing import List, Optional, Union, Tuple, TypedDict, Iterator, Callable
import numpy as np
from sentence_transformers import SentenceTransformer
from jet.logger import logger
from jet.models.embeddings.base import generate_embeddings
from jet.models.model_registry.transformers.sentence_transformer_registry import SentenceTransformerRegistry
from jet.models.model_types import EmbedModelType
from jet.transformers.formatters import format_json
from jet.wordnet.text_chunker import chunk_texts_with_data

DEFAULT_EMBED_MODEL: EmbedModelType = 'static-retrieval-mrl-en-v1'
MAX_CONTENT_SIZE = 1000


class TextSearchMetadata(TypedDict):
    """Typed dictionary for search result metadata."""
    text_id: str
    start_idx: int
    end_idx: int
    chunk_idx: int
    content_similarity: float
    num_tokens: int


class TextSearchResult(TypedDict):
    """Typed dictionary for search result structure."""
    rank: int
    score: float
    metadata: TextSearchMetadata
    text: str


class Weights(TypedDict):
    """Typed dictionary for similarity weights."""
    content: float


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    return dot_product / (norm_a * norm_b) if norm_a * norm_b != 0 else 0.0


def collect_text_chunks(
    texts: List[str],
    text_ids: Optional[List[str]] = None,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    tokenizer: Optional[Callable[[str], int]] = None
) -> Tuple[List[str], List[Tuple[str, str, int, int, int]]]:
    """
    Collect chunked contents for each text along with text IDs and token counts.

    Args:
        texts: List of text strings to process
        text_ids: Optional list of identifiers for each text
        chunk_size: Size of content chunks
        chunk_overlap: Overlap between chunks
        tokenizer: Optional callable to count tokens in text

    Returns:
        Tuple of (text_ids, contents_with_indices)
        where contents_with_indices = List of (text_id, content_chunk, start_idx, end_idx, num_tokens)
    """
    def default_tokenizer(text): return len(
        re.findall(r'\b\w+\b|[^\w\s]', text))
    tokenizer = tokenizer or default_tokenizer

    # Generate default text IDs if none provided
    text_ids = text_ids or [f"text_{i}" for i in range(len(texts))]
    if len(text_ids) != len(texts):
        raise ValueError("Number of text IDs must match number of texts")

    contents_with_indices = []
    for text, text_id in zip(texts, text_ids):
        chunks = chunk_texts_with_data(
            texts=[text],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            model=None,
            doc_ids=[text_id],
            buffer=0
        )
        for chunk in chunks:
            contents_with_indices.append(
                (
                    text_id,
                    chunk['content'],
                    chunk['start_idx'],
                    chunk['end_idx'],
                    chunk['num_tokens']
                )
            )

    return text_ids, contents_with_indices


def compute_weighted_similarity(
    query_vector: np.ndarray,
    content_vector: np.ndarray,
    weights: Optional[Weights] = None
) -> Tuple[float, float]:
    """
    Compute weighted similarity score for text content.

    Args:
        query_vector: Encoded query vector
        content_vector: Encoded content vector
        weights: Optional dictionary specifying weight for content similarity

    Returns:
        Tuple of (weighted_similarity, content_similarity)
    """
    content_sim = cosine_similarity(query_vector, content_vector)
    default_weights: Weights = {"content": 1.0}
    active_weights = weights if weights is not None else default_weights
    weighted_sim = active_weights["content"] * content_sim
    return weighted_sim, content_sim


def merge_results(
    results: List[TextSearchResult],
    tokenizer: Optional[Callable[[str], int]] = None
) -> List[TextSearchResult]:
    """
    Merge adjacent chunks from the same text into a single result, preserving order and metadata.

    Args:
        results: List of TextSearchResult dictionaries, potentially containing adjacent chunks
        tokenizer: Optional callable to count tokens in text

    Returns:
        List of TextSearchResult dictionaries with adjacent chunks merged
    """
    if not results:
        return []

    def default_tokenizer(text): return len(
        re.findall(r'\b\w+\b|[^\w\s]', text))
    tokenizer = tokenizer or default_tokenizer

    grouped: dict[str, List[TextSearchResult]] = {}
    for result in results:
        text_id = result["metadata"]["text_id"]
        if text_id not in grouped:
            grouped[text_id] = []
        grouped[text_id].append(result)

    merged_results: List[TextSearchResult] = []
    for text_id, chunks in grouped.items():
        chunks.sort(key=lambda x: x["metadata"]["start_idx"])
        current_chunk = chunks[0]
        merged_text = current_chunk["text"]
        start_idx = current_chunk["metadata"]["start_idx"]
        end_idx = current_chunk["metadata"]["end_idx"]
        max_score = current_chunk["score"]
        content_sims = [current_chunk["metadata"]["content_similarity"]]
        chunk_count = 1
        tokens = tokenizer(merged_text)

        for next_chunk in chunks[1:]:
            next_start = next_chunk["metadata"]["start_idx"]
            next_end = next_chunk["metadata"]["end_idx"]
            next_text = next_chunk["text"]
            if next_start <= end_idx:
                new_end = max(end_idx, next_end)
                overlap = end_idx - next_start
                additional_content = next_text[overlap:] if overlap > 0 else next_text
                merged_text += additional_content
                end_idx = new_end
                max_score = max(max_score, next_chunk["score"])
                content_sims.append(
                    next_chunk["metadata"]["content_similarity"])
                chunk_count += 1
                tokens = tokenizer(merged_text)
            else:
                avg_content_sim = sum(content_sims) / chunk_count
                merged_results.append({
                    "rank": current_chunk["rank"],
                    "score": max_score,
                    "metadata": {
                        "text_id": text_id,
                        "start_idx": start_idx,
                        "end_idx": end_idx,
                        "chunk_idx": 0,
                        "content_similarity": avg_content_sim,
                        "num_tokens": tokens
                    },
                    "text": merged_text,
                })
                current_chunk = next_chunk
                merged_text = current_chunk["text"]
                start_idx = current_chunk["metadata"]["start_idx"]
                end_idx = current_chunk["metadata"]["end_idx"]
                max_score = current_chunk["score"]
                content_sims = [current_chunk["metadata"]
                                ["content_similarity"]]
                chunk_count = 1
                tokens = tokenizer(merged_text)

        avg_content_sim = sum(content_sims) / chunk_count
        merged_results.append({
            "rank": current_chunk["rank"],
            "score": max_score,
            "metadata": {
                "text_id": text_id,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "chunk_idx": 0,
                "content_similarity": avg_content_sim,
                "num_tokens": tokens
            },
            "text": merged_text,
        })

    merged_results.sort(key=lambda x: x["score"], reverse=True)
    for i, result in enumerate(merged_results, 1):
        result["rank"] = i
    return merged_results


def search_texts(
    texts: List[str],
    query: str,
    text_ids: Optional[List[str]] = None,
    top_k: Optional[int] = None,
    embed_model: Union[SentenceTransformer,
                       EmbedModelType] = DEFAULT_EMBED_MODEL,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    threshold: float = 0.0,
    tokenizer: Optional[Callable[[str], int]] = None,
    split_chunks: bool = False,
    preprocess: Optional[Callable[[str], str]] = None,
    weights: Optional[Weights] = None
) -> Iterator[TextSearchResult]:
    """
    Search texts using vector similarity on chunked contents.
    Yields up to top_k results iteratively that meet the threshold, or all results if top_k is None.

    Args:
        texts: List of text strings to search
        query: Search query string
        text_ids: Optional list of identifiers for each text
        top_k: Maximum number of results to yield, or None to yield all results
        embed_model: Embedding model or model name to use for vectorization
        chunk_size: Size of content chunks
        chunk_overlap: Overlap between chunks
        threshold: Minimum similarity score for results
        tokenizer: Optional callable to count tokens in text
        split_chunks: If True, return individual chunks; if False, merge adjacent chunks
        preprocess: Optional callback to preprocess texts before embedding
        weights: Optional dictionary specifying weight for content similarity

    Returns:
        Iterator of TextSearchResult dictionaries (ranked by similarity)
    """
    def default_tokenizer(text): return len(
        re.findall(r'\b\w+\b|[^\w\s]', text))
    tokenizer = tokenizer or default_tokenizer

    text_ids, chunk_data = collect_text_chunks(
        texts, text_ids, chunk_size, chunk_overlap, tokenizer)
    logger.debug(f"Text IDs:\n\n{format_json(text_ids)}")

    if not chunk_data:
        return

    chunk_texts = [chunk for _, chunk, _, _, _ in chunk_data]
    processed_query = preprocess(query) if preprocess else query
    processed_chunk_texts = [preprocess(
        chunk) if preprocess else chunk for chunk in chunk_texts]

    logger.info(
        f"Generating embeddings for {len(processed_chunk_texts) + 1} texts:\n"
        f"  1 query\n"
        f"  {len(processed_chunk_texts)} chunks"
    )

    all_texts = [processed_query] + processed_chunk_texts
    all_vectors = generate_embeddings(
        all_texts,
        embed_model,
        return_format="numpy",
        batch_size=32,
        show_progress=True
    )

    query_vector = all_vectors[0]
    content_vectors = all_vectors[1:]

    results: List[TextSearchResult] = []
    chunk_counts = {}
    for i, (text_id, chunk, start_idx, end_idx, num_tokens) in enumerate(chunk_data):
        content_vector = content_vectors[i]
        weighted_sim, content_sim = compute_weighted_similarity(
            query_vector, content_vector, weights)

        if weighted_sim >= threshold:
            chunk_counts[text_id] = chunk_counts.get(text_id, -1) + 1
            result = {
                "rank": 0,
                "score": float(weighted_sim),
                "metadata": {
                    "text_id": text_id,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "chunk_idx": chunk_counts[text_id],
                    "content_similarity": float(content_sim),
                    "num_tokens": num_tokens
                },
                "text": chunk,
            }
            results.append(result)

    results.sort(key=lambda x: x["score"], reverse=True)
    if not split_chunks:
        results = merge_results(results, tokenizer)

    for i, result in enumerate(results if top_k is None else results[:top_k], 1):
        result["rank"] = i
        yield result
