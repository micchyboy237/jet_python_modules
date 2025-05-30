import pytest
from typing import List
from jet.llm.utils.bm25_plus import bm25_plus, SimilarityResult


class TestBM25Plus:
    def test_empty_corpus(self):
        corpus: List[str] = []
        query: str = "test query"
        expected: List[SimilarityResult] = []
        result: List[SimilarityResult] = bm25_plus(corpus, query)
        assert result == expected, "Empty corpus should return empty list"

    def test_empty_query(self):
        corpus: List[str] = ["doc one", "doc two"]
        query: str = ""
        expected: List[SimilarityResult] = [
            {"id": "doc_0", "rank": 1, "doc_index": 0,
                "score": 0.0, "text": "doc one", "tokens": 2},
            {"id": "doc_1", "rank": 2, "doc_index": 1,
                "score": 0.0, "text": "doc two", "tokens": 2}
        ]
        result: List[SimilarityResult] = bm25_plus(corpus, query)
        assert result == expected, "Empty query should return zero scores with sequential ranks"

    def test_unique_ranks_sorted_by_score(self):
        corpus: List[str] = ["apple banana", "banana cherry", "apple cherry"]
        query: str = "apple"
        result: List[SimilarityResult] = bm25_plus(corpus, query)
        expected_ranks: List[int] = [1, 2, 3]
        result_ranks: List[int] = [r["rank"] for r in result]
        result_scores: List[float] = [r["score"] for r in result]
        # Check ranks are unique and start from 1
        assert result_ranks == expected_ranks, "Ranks should be unique and start from 1"
        # Check scores are sorted in descending order
        assert result_scores == sorted(
            result_scores, reverse=True), "Results should be sorted by score descending"
        # Check higher score has lower rank
        for i in range(len(result) - 1):
            assert result[i]["score"] >= result[i +
                                                1]["score"], "Scores should be in descending order"

    def test_custom_doc_ids(self):
        corpus: List[str] = ["apple banana", "banana cherry"]
        query: str = "banana"
        doc_ids: List[str] = ["docA", "docB"]
        result: List[SimilarityResult] = bm25_plus(corpus, query, doc_ids)
        expected_ids: List[str] = ["docA", "docB"]
        result_ids: List[str] = [r["id"] for r in result]
        expected_ranks: List[int] = [1, 2]
        result_ranks: List[int] = [r["rank"] for r in result]
        assert result_ids == expected_ids, "Document IDs should match input"
        assert result_ranks == expected_ranks, "Ranks should be sequential starting from 1"

    def test_k1_diversity_spread(self):
        corpus: List[str] = ["apple apple banana",
                             "apple cherry", "banana cherry"]
        query: str = "apple"
        # Test with low k1 (less diversity)
        result_low_k1: List[SimilarityResult] = bm25_plus(
            corpus, query, k1=0.5)
        scores_low_k1: List[float] = [r["score"] for r in result_low_k1]
        score_range_low: float = max(scores_low_k1) - min(scores_low_k1)
        # Test with high k1 (more diversity)
        result_high_k1: List[SimilarityResult] = bm25_plus(
            corpus, query, k1=3.0)
        scores_high_k1: List[float] = [r["score"] for r in result_high_k1]
        score_range_high: float = max(scores_high_k1) - min(scores_high_k1)
        # Expect greater score spread with higher k1
        assert score_range_high > score_range_low, "Higher k1 should increase score diversity"
