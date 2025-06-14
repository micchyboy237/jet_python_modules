from typing import List, Optional
from jet.llm.utils.bm25_plus import bm25_plus, BM25PlusResult, bm25_plus_with_keyword_counts
from jet.logger import logger
from jet.wordnet.n_grams import count_ngrams


def rerank_bm25_plus(texts: List[str], query: str, keywords: Optional[dict[str, int]]) -> BM25PlusResult:
    """
    Reranks texts using BM25+ algorithm and returns top k results.

    Args:
        texts: List of texts to rerank
        top_k: Number of top results to return

    Returns:
        List of reranked texts limited to top_k
    """

    logger.info(f"Reranking texts ({len(texts)})...")

    # if keywords:
    #     bm25_plus_result = bm25_plus_with_keyword_counts(
    #         texts, keywords, query=query, k1=0.5, b=1.5, boost_factor=3.0)
    # else:
    bm25_plus_result = bm25_plus(texts, query)

    return bm25_plus_result
