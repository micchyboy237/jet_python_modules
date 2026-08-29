from typing import List

from jet.adapters.llama_cpp.config import EMBED_MODEL, RERANK_MODEL
from jet.adapters.llama_cpp.hybrid_utils import HybridSearchResult, hybrid_search
from jet.logger import logger


def rerank_extracted_content(
    query: str,
    content: str,
    top_n: int = 3,
    chunk_size: int = 500,
    overlap: int = 50,
    vector_threshold: float = 0.3,
) -> List[HybridSearchResult]:
    """
    Split extracted web content into chunks and rerank against query.

    Uses hybrid_search (vector + cross-encoder) for precise relevance.
    Falls back gracefully if rerank server is unavailable.
    """
    if not content or not query:
        return []

    # Simple chunking; replace with token-aware splitter if needed
    chunks = []
    start = 0
    while start < len(content):
        end = min(start + chunk_size, len(content))
        chunks.append(content[start:end])
        start += chunk_size - overlap

    if not chunks:
        return []

    logger.debug(f"Reranking {len(chunks)} chunks for query: {query[:60]}...")

    try:
        results = hybrid_search(
            query=query,
            documents=chunks,
            top_n=top_n,
            vector_score_threshold=vector_threshold,
            embed_model=EMBED_MODEL,
            rerank_model=RERANK_MODEL,
            normalize_scores=True,
        )
        logger.info(f"Reranked {len(chunks)} chunks → {len(results)} results")
        return results
    except Exception as e:
        logger.warning(f"Hybrid rerank failed, returning empty: {e}")
        return []
