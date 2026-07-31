"""
Hybrid Search Utilities
Combines fast vector search (cosine similarity) with precise cross-encoder
reranking for high-quality retrieval. The two-stage pipeline:
  1. Vector search retrieves top_k candidates (k > final_n)
  2. Reranker scores and reorders candidates, returning top_n

Environment Variables:
  EMBED_QUERY_PREFIX  - Prefix added to queries before embedding (optional)
  EMBED_DOC_PREFIX    - Prefix added to documents before embedding (optional)
"""

import os
from typing import TypedDict

import numpy as np
from jet.adapters.llama_cpp.embed_utils import embed
from jet.adapters.llama_cpp.rerank_utils import rerank
from jet.adapters.llama_cpp.scoring_utils import cosine_similarity
from jet.logger import logger


class HybridSearchResult(TypedDict):
    """A single result from hybrid_search."""

    rank: int
    index: int
    score: float  # Normalized score (0–1) for interpretability
    text: str
    vector_score: float  # Original vector similarity score
    rerank_score_raw: float  # Raw reranker score (for debugging)


def _sigmoid_normalize_scores(
    scores: list[float],
    temperature: float = 1.0,
) -> list[float]:
    """
    Apply sigmoid normalization to convert raw scores to 0–1 range.

    Uses sigmoid function: normalized = 1 / (1 + exp(-score / temperature))

    Args:
        scores: List of raw scores (can be negative or positive).
        temperature: Controls the steepness of the sigmoid curve.
                     Lower = more extreme (closer to 0 or 1).
                     Higher = more moderate (closer to 0.5).
                     Default 1.0 works well for most cross-encoders.

    Returns:
        List of normalized scores in range (0, 1).
    """
    if not scores:
        return []

    normalized = []
    for score in scores:
        # Apply sigmoid with temperature scaling
        norm_score = 1.0 / (1.0 + np.exp(-score / temperature))
        normalized.append(float(norm_score))

    return normalized


def hybrid_search(
    query: str,
    documents: list[str],
    top_n: int = 5,
    vector_k: int | None = None,
    query_prefix: str | None = None,
    document_prefix: str | None = None,
    normalize_scores: bool = True,
    sigmoid_temperature: float = 1.0,
) -> list[HybridSearchResult]:
    """
    Two-stage hybrid search: vector retrieval → cross-encoder reranking.

    Stage 1 (Vector Search):
      Embeds query and documents, computes cosine similarity, and selects
      the top vector_k candidates. This is fast but less precise.

    Stage 2 (Reranking):
      Passes only the vector_k candidates through a cross-encoder reranker
      model for precise relevance scoring. Returns top_n final results.

      Scores are normalized to 0–1 range using sigmoid by default for better
      interpretability.

    Args:
        query: Search query string.
        documents: List of document strings to search over.
        top_n: Number of final results to return (default: 5).
        vector_k: Number of candidates from vector search to pass to reranker.
                  Defaults to max(top_n * 3, 10), capped at len(documents).
                  A larger value improves recall at the cost of reranking time.
        query_prefix: Optional prefix for query embedding (e.g., "search_query: ").
                      Falls back to EMBED_QUERY_PREFIX env var.
        document_prefix: Optional prefix for document embeddings (e.g., "search_document: ").
                         Falls back to EMBED_DOC_PREFIX env var.
        normalize_scores: If True (default), apply sigmoid normalization to rerank scores.
        sigmoid_temperature: Controls sigmoid steepness (default: 1.0).
                            Lower = more extreme, Higher = more moderate.

    Returns:
        List of HybridSearchResult dicts sorted by reranker score (descending).
        Each result includes:
          - rank: Final rank after reranking
          - index: Original index in documents list
          - score: Normalized reranker score (0–1, higher = more relevant)
          - text: Document text
          - vector_score: Original vector similarity score (for comparison)
          - rerank_score_raw: Raw reranker score before normalization

    Example:
        >>> results = hybrid_search("What is a panda?", docs, top_n=3)
        >>> for r in results:
        ...     print(f"#{r['rank']} (vector={r['vector_score']:.4f}, score={r['score']:.4f}): {r['text']}")
    """
    if not documents:
        logger.warning("hybrid_search: empty documents list, returning []")
        return []

    n_docs = len(documents)
    if vector_k is None:
        vector_k = max(top_n * 3, 10)
    vector_k = min(vector_k, n_docs)

    logger.info(
        f"hybrid_search: query='{query[:80]}...', "
        f"n_docs={n_docs}, top_n={top_n}, vector_k={vector_k}"
    )

    resolved_query_prefix = (
        query_prefix
        if query_prefix is not None
        else os.getenv("EMBED_QUERY_PREFIX") or None
    )
    resolved_doc_prefix = (
        document_prefix
        if document_prefix is not None
        else os.getenv("EMBED_DOC_PREFIX") or None
    )

    # Stage 1: Vector search
    logger.info("Stage 1/2: Vector search (embedding + cosine similarity)")
    query_emb = embed(query, prefix=resolved_query_prefix)
    doc_embs = embed(documents, prefix=resolved_doc_prefix)

    similarities = np.array(
        [cosine_similarity(query_emb, doc_emb) for doc_emb in doc_embs]
    )

    # Get all indices sorted by vector score (descending)
    all_sorted_indices = np.argsort(similarities)[::-1].tolist()

    # Select top vector_k candidates
    if vector_k >= n_docs:
        candidate_indices = list(range(n_docs))
    else:
        candidate_indices = all_sorted_indices[:vector_k]

    candidates = [documents[i] for i in candidate_indices]
    candidate_vector_scores = [float(similarities[i]) for i in candidate_indices]

    logger.info(
        f"Vector search complete: {len(candidates)} candidates selected from {n_docs} documents"
    )
    logger.debug(
        "Vector search results (before reranking): "
        + ", ".join(
            [
                f"idx={idx}(vec={score:.4f})"
                for idx, score in zip(candidate_indices, candidate_vector_scores)
            ]
        )
    )
    logger.debug(
        f"Vector score stats: min={min(candidate_vector_scores):.4f}, "
        f"max={max(candidate_vector_scores):.4f}, "
        f"mean={np.mean(candidate_vector_scores):.4f}"
    )

    # Stage 2: Reranking
    logger.info("Stage 2/2: Cross-encoder reranking")
    rerank_results = rerank(query, candidates, top_n=min(top_n, len(candidates)))

    # Extract raw rerank scores for normalization
    raw_rerank_scores = [rr["score"] for rr in rerank_results]

    # Normalize scores if requested
    if normalize_scores and raw_rerank_scores:
        normalized_scores = _sigmoid_normalize_scores(
            raw_rerank_scores,
            temperature=sigmoid_temperature,
        )
        logger.info(
            f"Score normalization applied (sigmoid, temperature={sigmoid_temperature})"
        )
        logger.debug(
            f"Raw scores: [{', '.join(f'{s:.4f}' for s in raw_rerank_scores)}]"
        )
        logger.debug(
            f"Normalized scores: [{', '.join(f'{s:.4f}' for s in normalized_scores)}]"
        )
    else:
        # If no normalization, use raw scores as-is
        normalized_scores = raw_rerank_scores
        if raw_rerank_scores:
            logger.info("Score normalization disabled, using raw scores")

    # Build final results with both vector and rerank scores
    final_results: list[HybridSearchResult] = []
    for rank_pos, rr in enumerate(rerank_results, start=1):
        # rr["index"] is the index into the candidates list
        candidate_local_idx = rr["index"]
        original_idx = candidate_indices[candidate_local_idx]
        vector_score = candidate_vector_scores[candidate_local_idx]

        # Use the score directly from rerank_results (already in correct order)
        raw_score = rr["score"]

        # Find the corresponding normalized score by matching position in rerank_results
        norm_score = normalized_scores[rank_pos - 1]

        final_results.append(
            {
                "rank": rank_pos,
                "index": original_idx,
                "score": norm_score,
                "text": documents[original_idx],
                "vector_score": vector_score,
                "rerank_score_raw": raw_score,
            }
        )

    # Log rerank score statistics for interpretability
    logger.info(f"Reranking complete: {len(final_results)} final results returned")
    logger.debug(
        "Normalized score interpretation: "
        "0.5 = neutral, >0.7 = high relevance, <0.3 = low relevance. "
        "Relative ordering matters more than absolute values."
    )
    logger.debug(
        f"Normalized score stats: min={min(normalized_scores):.4f}, "
        f"max={max(normalized_scores):.4f}, "
        f"mean={np.mean(normalized_scores):.4f}"
    )

    # Log comparison between vector and rerank ordering
    logger.debug(
        "Score comparison (vector → rerank normalized): "
        + ", ".join(
            [
                f"#{r['rank']} idx={r['index']} (vec={r['vector_score']:.4f} → score={r['score']:.4f})"
                for r in final_results
            ]
        )
    )

    return final_results


if __name__ == "__main__":
    query = "What is a giant panda?"
    docs = [
        "The giant panda is a bear species endemic to China.",
        "Python is a high-level programming language.",
        "Bears are carnivoran mammals of the family Ursidae.",
        "Machine learning is a subset of artificial intelligence.",
        "Pandas eat bamboo and live in mountainous regions.",
    ]

    print("=" * 60)
    print("HYBRID SEARCH RESULTS (CLEAN)")
    print("=" * 60)
    results = hybrid_search(query, docs, top_n=3, normalize_scores=True)

    print(f"\nQuery: {query}\n")
    print("Final ranked results (after reranking):")
    print("Format: #rank  idx  score(0-1)  vector  raw  text")
    print("-" * 60)
    for r in results:
        print(
            f"  #{r['rank']}  idx={r['index']}  "
            f"score={r['score']:.4f}  vector={r['vector_score']:.4f}  raw={r['rerank_score_raw']:.4f}  "
            f"{r['text']}"
        )
