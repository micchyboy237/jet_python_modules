import os
from jet.file.utils import load_file

from jet.search.similarity import SimilarityResult, get_bm25_similarities
from jet.search.transformers import clean_string
from typing import List, Dict, Any, Optional, TypedDict
from jet.transformers.formatters import format_json
from jet.wordnet.n_grams import count_ngrams, extract_ngrams, get_most_common_ngrams
from jet.wordnet.words import count_words, get_words
from shared.data_types.job import JobData
from jet.cache.cache_manager import CacheManager

cache_manager = CacheManager()


class SimilarityRequestData(TypedDict):
    queries: List[str]
    data_file: str


# class SimilarityDataItem(TypedDict):
#     id: str
#     score: float
#     similarity: Optional[float]
#     words: int
#     matched: dict[str, int]
#     text: str


class SimilarityResultData(TypedDict):
    queries: list[str]
    count: int
    matched: dict[str, int]
    data: List[SimilarityResult]


def rerank_bm25(queries: list[str], sentences: list[str], ids: list[str]) -> SimilarityResultData:
    """Processes BM25+ similarity search by handling cache, cleaning data, generating n-grams, and computing similarities."""

    similarity_results = get_bm25_similarities(queries, sentences, ids)

    # Filter out results without matches
    similarity_results = [
        result for result in similarity_results if result["matched"]]

    # Aggregate all "matched"
    matched = {query.lower(): 0 for query in queries}
    for result in similarity_results:
        result_matched = result["matched"]
        for match_query, match in result_matched.items():
            matched[match_query] += 1

    return {
        "queries": queries,
        "count": len(similarity_results),
        "matched": matched,
        "data": similarity_results,
    }
