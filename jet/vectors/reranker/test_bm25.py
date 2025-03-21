import unittest
from jet.vectors.reranker.bm25 import rerank_bm25, adjust_score_with_rewards_and_penalties
from jet.cache.cache_manager import CacheManager


class TestRerankBM25(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialize a single shared CacheManager instance for all tests."""
        cls.cache_manager = CacheManager()
        cls.cache_manager.invalidate_cache()  # Ensure fresh state

    def setUp(self):
        """Reset test data before each test."""
        self.sentences = [
            "The quick brown fox jumps over the lazy dog",
            "A fast, agile fox jumps high",
            "The brown fox and the lazy dog are friends",
        ]
        self.ids = ["0", "1", "2"]

    def test_basic_ranking(self):
        queries = ["quick brown fox", "lazy dog"]
        result = rerank_bm25(queries, self.sentences, self.ids)

        self.assertEqual(result["data"][0]["text"],
                         "The quick brown fox jumps over the lazy dog")
        self.assertEqual(result["data"][1]["text"],
                         "The brown fox and the lazy dog are friends")

    def test_cache_updates_correctly(self):
        queries = ["fox"]

        # First call should populate cache
        self.cache_manager.invalidate_cache()
        rerank_bm25(queries, self.sentences, self.ids)
        self.assertFalse(self.cache_manager.is_cache_valid())

        # Second call should reuse cache without modification
        old_cache = self.cache_manager.cache.copy()
        rerank_bm25(queries, self.sentences, self.ids)
        self.assertEqual(self.cache_manager.cache, old_cache)

    def test_empty_queries(self):
        result = rerank_bm25([], self.sentences, self.ids)
        self.assertEqual(result["count"], 0)
        self.assertEqual(result["data"], [])

    def test_empty_sentences(self):
        with self.assertRaises(ValueError):
            rerank_bm25(["fox"], [], [])


class TestScoreAdjustments(unittest.TestCase):
    def test_full_match_boost(self):
        """Test when all queries match, the score gets the max boost (50%)."""
        base_score = 1.0
        match_count = 10  # Assume 10 matches
        max_query_count = 10  # Total 10 queries
        expected = base_score * 1.5  # 50% boost
        result = adjust_score_with_rewards_and_penalties(
            base_score, match_count, max_query_count)
        self.assertAlmostEqual(result, expected, places=5)

    def test_no_match_penalty(self):
        """Test when no queries match, the score gets the max penalty (30%)."""
        base_score = 1.0
        match_count = 0  # No matches
        max_query_count = 10  # Total 10 queries
        expected = base_score * 0.7  # 30% penalty
        result = adjust_score_with_rewards_and_penalties(
            base_score, match_count, max_query_count)
        self.assertAlmostEqual(result, expected, places=5)

    def test_half_match_adjustment(self):
        """Test when half of the queries match, the adjustment is balanced."""
        base_score = 1.0
        match_count = 5  # Half the queries match
        max_query_count = 10  # Total 10 queries
        # 25% boost - 15% penalty
        expected = base_score * (1 + (0.5 * 0.5) - (0.5 * 0.3))
        result = adjust_score_with_rewards_and_penalties(
            base_score, match_count, max_query_count)
        self.assertAlmostEqual(result, expected, places=5)

    def test_single_match_adjustment(self):
        """Test when only 1 query matches out of many, ensuring a small boost."""
        base_score = 1.0
        match_count = 1
        max_query_count = 10
        # Minor boost, more penalty
        expected = base_score * (1 + (1/10 * 0.5) - (9/10 * 0.3))
        result = adjust_score_with_rewards_and_penalties(
            base_score, match_count, max_query_count)
        self.assertAlmostEqual(result, expected, places=5)

    def test_no_queries_edge_case(self):
        """Test when there are no queries to avoid division by zero errors."""
        base_score = 1.0
        match_count = 0
        max_query_count = 0  # No queries at all
        expected = base_score  # Should return the original score
        result = adjust_score_with_rewards_and_penalties(
            base_score, match_count, max_query_count)
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
