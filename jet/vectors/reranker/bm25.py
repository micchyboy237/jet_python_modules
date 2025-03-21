import os
from jet.file.utils import load_file

from jet.search.similarity import BM25SimilarityResult, get_bm25_similarities
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


class SimilarityDataItem(TypedDict):
    id: str
    score: float
    similarity: Optional[float]
    words: int
    matched: dict[str, int]
    text: str


class SimilarityResultData(TypedDict):
    count: int
    matched: dict[str, int]
    data: List[SimilarityDataItem]


def adjust_score_with_rewards_and_penalties(base_score: float, match_count: int, max_query_count: int) -> float:
    """
    Adjusts the score based on query match count using rewards and penalties.

    - Rewards: Boost score if more queries match (up to 50% boost).
    - Penalties: Reduce score if fewer queries match (up to 30% penalty).

    :param base_score: Original BM25 similarity score.
    :param match_count: Number of matched n-grams.
    :param max_query_count: Total number of queries.
    :return: Adjusted similarity score.
    """
    if max_query_count == 0:
        return base_score  # Avoid division by zero

    boost_factor = (match_count / max_query_count) * 0.5  # Max 50% boost
    penalty_factor = (1 - match_count / max_query_count) * \
        0.3  # Max 30% penalty

    return base_score * (1 + boost_factor - penalty_factor)


def rerank_bm25(queries: list[str], sentences: list[str], ids: list[str]) -> SimilarityResultData:
    """Processes BM25+ similarity search by handling cache, cleaning data, generating n-grams, and computing similarities."""

    # Lowercase
    queries = [text.lower() for text in queries]
    sentences = [text.lower() for text in sentences]

    if not queries or not sentences or not ids:
        raise ValueError("All inputs most not be empty")

    # Load previous cache data
    cache_data: Dict[str, Any] = cache_manager.cache

    # Check if cache is valid
    if not cache_manager.is_cache_valid():
        # Generate n-grams
        common_texts_ngrams: List[List[str]] = [
            list(count_ngrams(sentence, min_count=1, max_words=5).keys()) for sentence in sentences
        ]

        # Update cache
        cache_data = cache_manager.update_cache(common_texts_ngrams)
    else:
        common_texts_ngrams: List[List[str]
                                  ] = cache_data["common_texts_ngrams"]

    # Preprocess queries
    formatted_queries: List[str] = [
        " ".join(text.split())
        for text in queries
    ]

    common_texts: List[str] = [" ".join(
        [" ".join(text.split()) for text in texts]) for texts in common_texts_ngrams]

    # Compute BM25+ similarities
    similarities: List[BM25SimilarityResult] = get_bm25_similarities(
        formatted_queries, common_texts, ids)

    # Rerank with rewards/penalties
    max_query_count = len(queries)
    results: List[SimilarityDataItem] = []

    for result in similarities:
        idx = int(result["id"])
        text = sentences[idx]

        # Count n-gram matches per query
        matched_ngrams = {
            ngram: count for ngram, count in count_ngrams(
                text, min_count=1, max_words=max(count_words(text) for text in formatted_queries)
            ).items()
            if ngram in formatted_queries
        }

        match_count = len(matched_ngrams)  # Number of matched n-grams
        adjusted_score = adjust_score_with_rewards_and_penalties(
            result["score"], match_count, max_query_count
        )

        results.append({
            "id": result["id"],
            "score": adjusted_score,
            "similarity": result["similarity"] or None,
            "words": count_words(text),
            "matched": matched_ngrams,
            "text": text,
        })

    # Sort by adjusted scores (higher is better)
    results.sort(key=lambda x: x["score"], reverse=True)

    # Aggregate all "matched"
    matched = {query: 0 for query in queries}
    for result in results:
        result_matched = result["matched"]
        for match, count in result_matched.items():
            matched[match] += count

    return {"count": len(results), "matched": matched, "data": results}
