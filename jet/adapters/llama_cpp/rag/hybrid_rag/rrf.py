import logging
from collections import defaultdict
from typing import Dict, List, Optional

from .models import Document, RetrievalResult

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    results: List[RetrievalResult],
    k: int = 60,
    weights: Optional[Dict[str, float]] = None,
    top_n: Optional[int] = None,
) -> List[Document]:
    """
    Fuse multiple ranked lists using Reciprocal Rank Fusion.

    Args:
        results: List of RetrievalResult (one per retriever)
        k: Smoothing constant (default 60 – industry standard)
        weights: Optional dict {retriever_name: weight}
        top_n: Keep only top N after fusion (None = keep all)

    Returns:
        Sorted list of Documents by descending RRF score
    """
    if not results:
        logger.warning("No retrieval results provided to RRF")
        return []

    weights = weights or {}
    score_map: Dict[str, float] = defaultdict(float)
    doc_map: Dict[str, Document] = {}

    for result in results:
        weight = weights.get(result.retriever_name, 1.0)
        logger.debug(
            f"Processing {result.retriever_name} "
            f"({len(result.documents)} docs, weight={weight})"
        )

        for rank, doc in enumerate(result.documents, start=1):
            # rank is 1-based
            contribution = weight * (1.0 / (k + rank))
            score_map[doc.id] += contribution

            # Keep the first occurrence of the document object
            if doc.id not in doc_map:
                doc_map[doc.id] = doc

    # Attach RRF scores and sort
    fused: List[Document] = []
    for doc_id, rrf_score in score_map.items():
        doc = doc_map[doc_id]
        doc.rrf_score = rrf_score
        fused.append(doc)

    fused.sort(key=lambda d: d.rrf_score, reverse=True)

    if top_n is not None:
        fused = fused[:top_n]

    logger.info(
        f"RRF fused {len(score_map)} unique documents → returning top {len(fused)}"
    )
    return fused
