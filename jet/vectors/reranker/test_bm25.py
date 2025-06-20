import pytest
from jet.vectors.reranker.bm25 import adjust_score_with_rewards_and_penalties, get_bm25_similarities, get_bm25_similarities_old, rerank_bm25
from jet.cache.cache_manager import CacheManager
from typing import List, Dict


@pytest.fixture(scope="class")
def cache_manager():
    """Initialize a single shared CacheManager instance for all tests."""
    cache_manager = CacheManager()
    cache_manager.invalidate_cache()
    return cache_manager


@pytest.fixture
def test_data():
    """Provide test data for each test."""
    sentences = [
        "The quick brown fox jumps over the lazy dog",
        "A fast, agile fox jumps high",
        "The brown fox and the lazy dog are friends",
    ]
    ids = ["0", "1", "2"]
    return sentences, ids


class TestRerankBM25:
    def test_basic_ranking(self, test_data):
        """Given queries and sentences, when rerank_bm25 is called, then results are ranked correctly."""
        sentences, ids = test_data
        queries = ["quick brown fox", "lazy dog"]
        expected_top_text = "The quick brown fox jumps over the lazy dog"
        expected_second_text = "The brown fox and the lazy dog are friends"

        result = rerank_bm25(queries, sentences, ids)

        assert result["data"][0]["text"] == expected_top_text, "Top result should match the most relevant sentence."
        assert result["data"][1]["text"] == expected_second_text, "Second result should match the next relevant sentence."

    def test_cache_updates_correctly(self, test_data, cache_manager):
        """Given a query and cache, when rerank_bm25 is called twice, then cache remains consistent."""
        sentences, ids = test_data
        queries = ["fox"]

        cache_manager.invalidate_cache()
        result_first = rerank_bm25(queries, sentences, ids)
        old_cache = cache_manager.cache.copy()
        result_second = rerank_bm25(queries, sentences, ids)

        assert cache_manager.cache == old_cache, "Cache should remain consistent between calls."
        assert not cache_manager.is_cache_valid(
        ), "Cache should be invalidated after first call."

    def test_empty_queries(self, test_data):
        """Given empty queries, when rerank_bm25 is called, then it returns empty results."""
        sentences, ids = test_data
        queries = []
        expected = {
            "queries": [],
            "count": 0,
            "matched": {},
            "data": [],
        }

        result = rerank_bm25(queries, sentences, ids)

        assert result == expected, "Empty queries should return empty results."

    def test_empty_sentences(self, test_data):
        """Given empty sentences, when rerank_bm25 is called, then it raises ValueError."""
        _, ids = test_data
        queries = ["fox"]

        with pytest.raises(ValueError, match="queries and documents must not be empty"):
            rerank_bm25(queries, [], [])


class TestScoreAdjustments:
    def test_full_match_boost(self):
        """Given all query terms matched, when adjust_score_with_rewards_and_penalties is called, then score is boosted by 50%."""
        base_score = 1.0
        matched_terms = {"term1": 1, "term2": 1}
        query_terms = ["term1", "term2"]
        idf = {"term1": 1.0, "term2": 1.0}
        # Reward: 2 * 0.8, Penalty: 0
        expected = base_score * (1 + (2 * 0.8)) * (1 - 0)

        result = adjust_score_with_rewards_and_penalties(
            base_score, matched_terms, query_terms, idf)

        assert result == pytest.approx(
            expected, abs=1e-5), "Full match should boost score by 50%."

    def test_no_match_penalty(self):
        """Given no query terms matched, when adjust_score_with_rewards_and_penalties is called, then score is penalized."""
        base_score = 1.0
        matched_terms = {}
        query_terms = ["term1", "term2"]
        idf = {"term1": 1.0, "term2": 1.0}
        penalty = math.log1p(2) / math.log1p(2)  # 2 missing terms
        expected = base_score * (1 + 0) * (1 - penalty)

        result = adjust_score_with_rewards_and_penalties(
            base_score, matched_terms, query_terms, idf)

        assert result == pytest.approx(
            expected, abs=1e-5), "No matches should apply max penalty."

    def test_half_match_adjustment(self):
        """Given half of query terms matched, when adjust_score_with_rewards_and_penalties is called, then score is adjusted accordingly."""
        base_score = 1.0
        matched_terms = {"term1": 1}
        query_terms = ["term1", "term2"]
        idf = {"term1": 1.0, "term2": 1.0}
        reward = 1 * 0.8
        penalty = math.log1p(1) / math.log1p(2)
        expected = base_score * (1 + reward) * (1 - penalty)

        result = adjust_score_with_rewards_and_penalties(
            base_score, matched_terms, query_terms, idf)

        assert result == pytest.approx(
            expected, abs=1e-5), "Half match should balance reward and penalty."

    def test_single_match_adjustment(self):
        """Given one query term matched out of many, when adjust_score_with_rewards_and_penalties is called, then score has small boost."""
        base_score = 1.0
        matched_terms = {"term1": 1}
        query_terms = ["term1", "term2", "term3", "term4"]
        idf = {"term1": 1.0}
        reward = 1 * 0.8
        penalty = math.log1p(3) / math.log1p(4)
        expected = base_score * (1 + reward) * (1 - penalty)

        result = adjust_score_with_rewards_and_penalties(
            base_score, matched_terms, query_terms, idf)

        assert result == pytest.approx(
            expected, abs=1e-5), "Single match should apply small boost with penalty."

    def test_no_queries_edge_case(self):
        """Given no query terms, when adjust_score_with_rewards_and_penalties is called, then base score is returned."""
        base_score = 1.0
        matched_terms = {}
        query_terms = []
        idf = {}
        expected = base_score

        result = adjust_score_with_rewards_and_penalties(
            base_score, matched_terms, query_terms, idf)

        assert result == expected, "No queries should return base score."


class TestBM25SimilarityRanking:
    @pytest.fixture
    def ranking_data(self):
        """Provide diverse queries and documents for ranking tests."""
        queries = [
            "apple",
            "banana smoothie",
            "cherry tart",
            "fresh orange juice",
            "grape",
            "blueberry pancakes",
            "healthy breakfast options"
        ]
        documents = [
            "I love apple pie and banana bread. Apple pie is my favorite dessert.",
            "Cherry tart and grape jelly are my favorite desserts. Cherry tart is so delicious!",
            "A fresh orange juice with a banana smoothie is refreshing. Fresh orange juice is a must-have every morning.",
            "Apple pie pairs well with fresh orange juice and blueberry pancakes. I always have fresh orange juice.",
            "Grape jelly is delicious on toast with butter. Grape jelly and apple pie make a great pairing.",
            "Healthy breakfast options often include oatmeal, fruit, and smoothies. A banana smoothie is a great option.",
            "Blueberry pancakes are best with maple syrup. Blueberry pancakes with a banana smoothie are my go-to.",
            "A smoothie with banana, apple, and blueberries is a great start to the day. A banana smoothie is a quick breakfast.",
            "Orange juice and toast make a simple breakfast. Fresh orange juice is great with banana bread.",
            "I prefer a cherry tart over an apple pie any day. Cherry tart is simply the best dessert.",
            "Grapes and cherries are rich in antioxidants. A fresh cherry tart is a great treat.",
            "Pancakes with bananas and blueberries are a great healthy breakfast option. Blueberry pancakes are always a weekend favorite.",
            "This document has nothing to do with food.",
            "Banana smoothies and apple pie are perfect for a summer day. A banana smoothie is so refreshing.",
            "Fresh orange juice, blueberry pancakes, and a cherry tart make a great breakfast. Fresh orange juice is a staple for me.",
            "I often eat a healthy breakfast with an apple, banana smoothie, and grape jelly toast. Healthy breakfast options always include a banana smoothie.",
            "Blueberry pancakes and fresh orange juice are my weekend treats. Blueberry pancakes are amazing with banana bread.",
            "Healthy breakfast options include fresh fruit, cherry tarts, and banana smoothies. A cherry tart can also be a healthy choice.",
            "Apple pie and blueberry pancakes with fresh orange juice are my favorite morning meal. Fresh orange juice is always part of my morning routine.",
            "Grapes, bananas, and oranges are a great mix for a smoothie. A banana smoothie with fresh orange juice is a power drink.",
            "For a delicious breakfast, try a cherry tart with grape jelly toast and fresh orange juice. A fresh cherry tart always makes breakfast better.",
            "A smoothie with banana, apple, and orange juice is a nutritious choice. Banana smoothie lovers should try it with fresh orange juice.",
            "Blueberry pancakes with a cherry tart on the side make a perfect meal. Cherry tart and blueberry pancakes complement each other well."
        ]
        return queries, documents

    def test_top_results_are_diverse(self, ranking_data):
        """Given diverse queries and documents, when get_bm25_similarities is called, then top results match multiple queries."""
        queries, documents = ranking_data
        expected_min_queries = 3

        results = get_bm25_similarities(queries, documents)
        top_result = results[0]
        second_result = results[1] if len(results) > 1 else None
        unique_queries_covered = set()
        for result in results[:3]:
            unique_queries_covered.update(result["matched"].keys())

        assert len(top_result["matched"]
                   ) > 1, "Top result should match multiple queries."
        if second_result:
            assert len(
                second_result["matched"]) > 0, "Second result should have at least one query match."
        assert len(
            unique_queries_covered) >= expected_min_queries, "Top 3 results should match at least 3 unique queries."

    def test_results_are_sorted_correctly(self, ranking_data):
        """Given queries and documents, when get_bm25_similarities is called, then results are sorted by score."""
        queries, documents = ranking_data

        results = get_bm25_similarities(queries, documents)
        scores = [entry["score"] for entry in results]

        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1)
                   ), "Results should be sorted by descending score."

    def test_results_contain_expected_queries(self, ranking_data):
        """Given queries and documents, when get_bm25_similarities is called, then results contain expected query matches."""
        queries, documents = ranking_data

        results = get_bm25_similarities(queries, documents)

        for result in results[:3]:
            assert any(q in result["matched"]
                       for q in queries), f"Result '{result['text']}' should match at least one query."

    def test_low_relevance_documents_rank_lower(self, ranking_data):
        """Given queries and an unrelated document, when get_bm25_similarities_old is called, then unrelated document ranks lowest."""
        queries, documents = ranking_data
        extra_docs = documents + ["This document is completely unrelated."]
        expected_matches = 0
        expected_score = 0.0

        results = get_bm25_similarities_old(queries, extra_docs)
        last_result = results[-1]

        assert last_result["matched"] == {
        }, f"Unrelated document should have {expected_matches} query matches."
        assert last_result["score"] == pytest.approx(
            expected_score), "Unrelated document should have the lowest score."
