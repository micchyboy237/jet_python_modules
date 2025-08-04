import uuid
import numpy as np
from typing import List, Optional, TypedDict
from jet.logger import logger


class DiverseResult(TypedDict):
    id: str
    index: int
    text: str
    score: float


def select_mmr_texts(
    embeddings: np.ndarray,
    texts: List[str],
    query_embedding: np.ndarray,
    lambda_param: float = 0.5,
    max_texts: int = 5,
    ids: Optional[List[str]] = None
) -> List[DiverseResult]:
    """Select a diverse subset of texts using Maximum Marginal Relevance (MMR).

    Args:
        embeddings: Array of shape (n_texts, embedding_dim) with text embeddings.
        texts: List of texts corresponding to embeddings.
        query_embedding: Array of shape (embedding_dim,) representing the query.
        lambda_param: Trade-off between relevance and diversity (0 to 1).
        max_texts: Maximum number of texts to select.
        ids: Optional list of IDs for texts. If None, UUIDs are generated.

    Returns:
        List of DiverseResult dictionaries containing id, index, text, and MMR score.
    """
    logger.debug(
        f"Input embeddings shape: {embeddings.shape}, texts: {len(texts)}")
    logger.debug(
        f"Query embedding shape: {query_embedding.shape}, lambda: {lambda_param}, max_texts: {max_texts}")

    # Validate inputs
    if len(texts) == 0 or len(texts) != embeddings.shape[0]:
        logger.debug(
            "Empty texts or mismatched embeddings, returning empty list")
        return []
    if query_embedding.shape[0] != embeddings.shape[1]:
        logger.error(
            "Query embedding dimension does not match text embeddings")
        raise ValueError(
            "Query embedding dimension must match text embeddings")
    if not 0 <= lambda_param <= 1:
        logger.error("lambda_param must be between 0 and 1")
        raise ValueError("lambda_param must be between 0 and 1")
    if max_texts < 1:
        logger.error("max_texts must be at least 1")
        raise ValueError("max_texts must be at least 1")

    # Generate or validate IDs
    if ids is None:
        ids = [str(uuid.uuid4()) for _ in range(len(texts))]
    elif len(ids) != len(texts):
        logger.debug("Mismatched IDs length, generating new UUIDs")
        ids = [str(uuid.uuid4()) for _ in range(len(texts))]

    # Normalize embeddings for cosine similarity
    embeddings_norm = embeddings / \
        np.linalg.norm(embeddings, axis=1, keepdims=True)
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    logger.debug("Embeddings and query normalized")

    # Initialize with most relevant text
    relevance_scores = np.dot(embeddings_norm, query_norm).flatten()
    selected_indices = [int(np.argmax(relevance_scores))]
    results: List[DiverseResult] = [{
        "id": ids[selected_indices[0]],
        "index": selected_indices[0],
        "text": texts[selected_indices[0]],
        "score": float(relevance_scores[selected_indices[0]])
    }]
    remaining_indices = [i for i in range(
        len(texts)) if i not in selected_indices]
    logger.debug(
        f"Initial text selected: index {selected_indices[0]}, text: {texts[selected_indices[0]]}")

    # Iteratively select texts using MMR
    while len(results) < max_texts and remaining_indices:
        mmr_scores = []
        logger.debug(
            f"Iteration {len(results) + 1}, remaining indices: {remaining_indices}")
        for i in remaining_indices:
            # Relevance to query
            relevance = np.dot(embeddings_norm[i], query_norm)
            # Maximum similarity to selected texts
            max_similarity = max(
                np.dot(embeddings_norm[i], embeddings_norm[j]) for j in selected_indices)
            # MMR score
            mmr_score = lambda_param * relevance - \
                (1 - lambda_param) * max_similarity
            mmr_scores.append((mmr_score, i))
            logger.debug(
                f"Index {i}, text: {texts[i]}, relevance: {relevance:.6f}, max_similarity: {max_similarity:.6f}, mmr_score: {mmr_score:.6f}")

        if not mmr_scores:
            logger.debug("No remaining candidates, stopping")
            break

        # Select text with highest MMR score
        best_score, best_idx = max(mmr_scores)
        selected_indices.append(best_idx)
        results.append({
            "id": ids[best_idx],
            "index": best_idx,
            "text": texts[best_idx],
            "score": float(best_score)
        })
        remaining_indices.remove(best_idx)
        logger.debug(
            f"Selected text: index {best_idx}, text: {texts[best_idx]}, score: {best_score:.6f}")

    logger.debug(f"Final results: {[r['text'] for r in results]}")
    return results


__all__ = ["select_mmr_texts", "DiverseResult"]
