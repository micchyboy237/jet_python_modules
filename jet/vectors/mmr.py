from typing import List, TypedDict, Optional
import numpy as np
from jet.logger import logger


class MMRResult(TypedDict):
    index: int
    text: str
    similarity: float


def get_diverse_texts(
    query_embedding: np.ndarray,
    text_embeddings: np.ndarray,
    texts: List[str],
    mmr_lambda: float = 0.5,
    num_results: Optional[int] = None,
    initial_indices: Optional[List[int]] = None
) -> List[MMRResult]:
    """Select diverse and relevant texts using Maximum Marginal Relevance (MMR).

    Args:
        query_embedding: Normalized embedding of the query (shape: [1, dim]).
        text_embeddings: Normalized embeddings of texts (shape: [n, dim]).
        texts: List of text strings corresponding to embeddings.
        mmr_lambda: Weight for relevance vs. diversity (0 to 1). Higher favors relevance.
        num_results: Maximum number of results to return. If None, returns all available texts.
        initial_indices: Optional list of indices for pre-selected texts.

    Returns:
        List of MMRResult dictionaries with index, text, and similarity to query.

    Raises:
        ValueError: If inputs are invalid (empty query, mismatched shapes, or invalid indices).
    """
    # Validate inputs
    if query_embedding.size == 0:
        raise ValueError("Query embedding cannot be empty")
    if text_embeddings.shape[0] != len(texts):
        raise ValueError(
            "Number of text embeddings must match number of texts")
    if text_embeddings.size == 0 or len(texts) == 0:
        raise ValueError("Text embeddings and texts list cannot be empty")
    if not 0 <= mmr_lambda <= 1:
        raise ValueError("mmr_lambda must be between 0 and 1")
    if num_results is not None and num_results <= 0:
        raise ValueError("num_results must be positive")
    if initial_indices:
        if any(i < 0 or i >= len(texts) for i in initial_indices):
            raise ValueError("Initial indices out of range")
        if len(set(initial_indices)) != len(initial_indices):
            raise ValueError("Initial indices must be unique")

    # Initialize results and selected indices
    results: List[MMRResult] = []
    selected_indices = set(initial_indices) if initial_indices else set()
    available_indices = set(range(len(texts))) - selected_indices
    num_results = min(num_results or len(texts), len(texts))

    # Add initial indices to results if provided
    if initial_indices:
        query_similarities = np.dot(
            text_embeddings, query_embedding.T).flatten()
        for idx in initial_indices:
            results.append({
                "index": idx,
                "text": texts[idx],
                "similarity": float(query_similarities[idx])
            })

    # Compute initial similarities to query
    query_similarities = np.dot(text_embeddings, query_embedding.T).flatten()

    # MMR loop
    while len(results) < num_results and available_indices:
        max_mmr_score = float("-inf")
        best_idx = None

        # Compute MMR for each available text
        for idx in available_indices:
            # Relevance score (cosine similarity to query)
            relevance = query_similarities[idx]

            # Diversity score (minimum similarity to selected texts)
            diversity = float("inf")
            if selected_indices:
                selected_embeddings = text_embeddings[list(selected_indices)]
                similarities = np.dot(
                    selected_embeddings, text_embeddings[idx]).flatten()
                diversity = np.min(similarities)

            # MMR score
            mmr_score = mmr_lambda * relevance - (1 - mmr_lambda) * diversity

            if mmr_score > max_mmr_score:
                max_mmr_score = mmr_score
                best_idx = idx

        # Add best text to results
        if best_idx is not None:
            results.append({
                "index": best_idx,
                "text": texts[best_idx],
                "similarity": float(query_similarities[best_idx])
            })
            selected_indices.add(best_idx)
            available_indices.remove(best_idx)

    return results
