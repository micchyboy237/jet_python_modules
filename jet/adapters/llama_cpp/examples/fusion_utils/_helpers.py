"""Shared helpers for fusion_utils demos: convert ranked retrieval results
(from rerank_utils.rerank / vector_utils.vector_search) into the flat,
doc-index-ordered arrays that fusion_utils expects.
"""

from typing import Sequence, TypedDict


class IndexedResult(TypedDict, total=False):
    index: int
    score: float
    raw_score: float


def ranking_in_doc_order(results: Sequence[IndexedResult]) -> list[int]:
    """Extract the ranked list of original doc indices (best first)."""
    return [r["index"] for r in results]


def scores_in_doc_order(
    results: Sequence[IndexedResult],
    num_docs: int,
    score_key: str = "score",
) -> list[float]:
    """Re-project ranked results (sorted by relevance) back into a flat
    list aligned with the original document order, so fusion_utils can
    combine multiple signals index-for-index.
    """
    scores = [0.0] * num_docs
    for r in results:
        scores[r["index"]] = r[score_key]
    return scores
