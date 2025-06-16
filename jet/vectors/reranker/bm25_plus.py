from typing import List, Optional
from jet.llm.utils.bm25_plus import bm25_plus, bm25_plus_with_keyword_counts, BM25PlusResult
from jet.logger import logger


def rerank_bm25_plus(
    texts: List[str],
    query: str,
    *,
    keywords: Optional[dict[str, int]] = None,
    k1: float = 1.5,
    b: float = 0.75,
    delta: float = 1.0,
    boost_factor: float = 1.5
) -> BM25PlusResult:
    """
    Reranks texts using BM25+ algorithm and returns top k results.

    Args:
        texts: List of texts to rerank
        query: Search query string
        keywords: Optional dictionary mapping keywords to their counts
        k1: Controls term frequency impact on score (default: 1.5)
        b: Controls document length impact on score (default: 0.75)
        delta: Constant added to scores for non-negative values (default: 1.0)
        boost_factor: Factor to boost keyword matches (default: 1.5)

    Returns:
        BM25PlusResult containing ranked results and query term match counts

    Raises:
        ValueError: If texts is empty or query is empty
        TypeError: If texts contains non-string elements
        ValueError: If k1, b, delta, or boost_factor are non-positive
    """
    # Validate inputs
    if not texts:
        raise ValueError("Texts list cannot be empty")
    if not query:
        raise ValueError("Query string cannot be empty")
    if not all(isinstance(text, str) for text in texts):
        raise TypeError("All elements in texts must be strings")
    if k1 <= 0:
        raise ValueError("k1 must be positive")
    if b <= 0:
        raise ValueError("b must be positive")
    if delta <= 0:
        raise ValueError("delta must be positive")
    if boost_factor <= 0:
        raise ValueError("boost_factor must be positive")

    logger.info(f"Reranking texts ({len(texts)})...")

    if keywords:
        bm25_plus_result = bm25_plus_with_keyword_counts(
            texts,
            keywords,
            query=query,
            k1=k1,
            b=b,
            delta=delta,
            boost_factor=boost_factor
        )
    else:
        bm25_plus_result = bm25_plus(
            texts,
            query,
            k1=k1,
            b=b,
            delta=delta
        )

    return bm25_plus_result
