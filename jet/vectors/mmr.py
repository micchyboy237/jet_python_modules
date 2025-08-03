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
    initial_indices: List[int] = None
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
    if query_embedding.size == 0 or query_embedding.ndim != 2 or query_embedding.shape[0] != 1:
        raise ValueError(
            "Query embedding must be a 2D array with shape [1, dim]")
    if text_embeddings.ndim != 2:
        raise ValueError(
            "Text embeddings must be a 2D array with shape [n, dim]")
    if len(texts) != text_embeddings.shape[0]:
        raise ValueError(
            "Number of texts must match number of text embeddings")
    if not 0 <= mmr_lambda <= 1:
        raise ValueError("mmr_lambda must be between 0 and 1")
    if num_results is not None and num_results < 1:
        raise ValueError("num_results must be at least 1 if provided")

    num_texts = len(texts)
    if initial_indices is None:
        initial_indices = []
    else:
        if not all(isinstance(i, int) and 0 <= i < num_texts for i in initial_indices):
            raise ValueError(
                "initial_indices must contain valid indices within range")
        if len(set(initial_indices)) != len(initial_indices):
            raise ValueError("initial_indices must not contain duplicates")

    # Set num_results to all available texts if not provided
    num_results = num_texts if num_results is None else min(
        num_results, num_texts)
    if num_results == 0 or text_embeddings.shape[0] == 0:
        return []

    # Compute initial similarities to query
    similarities = np.dot(text_embeddings, query_embedding.T).flatten()
    logger.debug(f"Query similarities: {similarities}")
    logger.debug(
        f"Text embeddings norms: {np.linalg.norm(text_embeddings, axis=1)}")

    # Initialize result set with initial_indices
    selected_indices = list(initial_indices)
    available_indices = [i for i in range(
        num_texts) if i not in initial_indices]

    # Select first result (most relevant to query) if no initial indices
    if not selected_indices:
        if available_indices:
            first_idx = np.argmax(similarities[available_indices])
            first_idx = available_indices[first_idx]
            selected_indices.append(first_idx)
            available_indices.remove(first_idx)
            logger.debug(
                f"Selected first index {first_idx} ({texts[first_idx]}) with similarity {similarities[first_idx]:.3f}")

    # Select remaining results using MMR
    for iteration in range(num_results - len(selected_indices)):
        if not available_indices:
            break
        mmr_scores = np.zeros(len(available_indices))
        diversity_scores = np.zeros(len(available_indices))
        for i, idx in enumerate(available_indices):
            relevance = similarities[idx]
            if len(selected_indices) > 0:
                selected_embeddings = text_embeddings[selected_indices]
                diversity_matrix = np.dot(
                    text_embeddings[idx:idx+1], selected_embeddings.T).flatten()
                diversity = np.max(diversity_matrix)
                logger.debug(
                    f"Iteration {iteration}, Candidate {idx} diversity matrix: {diversity_matrix}")
            else:
                diversity = 0.0
            mmr_scores[i] = mmr_lambda * relevance - \
                (1 - mmr_lambda) * diversity
            diversity_scores[i] = diversity
            logger.debug(f"Iteration {iteration}, Candidate {idx} ({texts[idx]}): "
                         f"relevance={relevance:.3f}, diversity={diversity:.3f}, mmr_score={mmr_scores[i]:.3f}")
        max_score = np.max(mmr_scores)
        tie_threshold = 0.1
        max_indices = np.where(
            np.abs(mmr_scores - max_score) <= tie_threshold)[0]
        if len(max_indices) > 1:
            best_idx = available_indices[max_indices[np.argmin(
                diversity_scores[max_indices])]]
            logger.debug(f"Iteration {iteration}: Near-tie detected within {tie_threshold}, "
                         f"selected index {best_idx} ({texts[best_idx]}) with MMR score {mmr_scores[max_indices].max():.3f} "
                         f"and lowest diversity {diversity_scores[max_indices].min():.3f}")
        else:
            best_idx = available_indices[np.argmax(mmr_scores)]
            logger.debug(
                f"Iteration {iteration}: Selected index {best_idx} ({texts[best_idx]}) with MMR score {max_score:.3f}"
            )
        selected_indices.append(best_idx)
        available_indices.remove(best_idx)

    # Construct results
    results: List[MMRResult] = [
        {
            "index": idx,
            "text": texts[idx],
            "similarity": float(similarities[idx])
        }
        for idx in selected_indices
    ]

    logger.debug(f"Final selected indices: {selected_indices}")
    return results
