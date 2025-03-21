import unittest
from typing import List
from jet.search.similarity import get_bm25_similarities, adjust_score_with_rewards_and_penalties


class TestBM25Similarity(unittest.TestCase):

    def setUp(self):
        """ Set up sample queries and documents for testing """
        self.queries: List[str] = ["machine learning", "deep learning"]
        self.documents: List[str] = [
            "Machine learning is a method of data analysis.",
            "Deep learning is a subset of machine learning that uses neural networks.",
            "Neural networks are fundamental in deep learning."
        ]
        self.ids: List[str] = ["doc1", "doc2", "doc3"]

    def test_basic_matching(self):
        """ Test if BM25 correctly identifies relevant documents """
        results = get_bm25_similarities(self.queries, self.documents, self.ids)

        self.assertEqual(len(results), 3)  # Ensure all docs are included
        # Ensure sorting by score
        self.assertGreater(results[0]["score"], results[-1]["score"])

    def test_adjust_score_with_rewards_and_penalties(self):
        """ Test score adjustment function with various match counts """
        base_score = 1.0
        self.assertAlmostEqual(adjust_score_with_rewards_and_penalties(
            base_score, 2, 4), 1.1, places=2)
        self.assertAlmostEqual(adjust_score_with_rewards_and_penalties(
            base_score, 4, 4), 1.5, places=2)  # Max boost
        self.assertAlmostEqual(adjust_score_with_rewards_and_penalties(
            base_score, 0, 4), 0.7, places=2)  # Max penalty

    def test_partial_query_match(self):
        """ Test when only part of the query matches """
        partial_query = ["learning"]
        results = get_bm25_similarities(
            partial_query, self.documents[:len(partial_query)], self.ids[:len(partial_query)])

        # At least one doc should have nonzero score
        self.assertTrue(any(res["score"] > 0 for res in results))

    def test_no_match_case(self):
        """ Test when queries do not match any documents """
        no_match_query = ["quantum computing"]
        results = get_bm25_similarities(
            no_match_query, self.documents[:len(no_match_query)], self.ids[:len(no_match_query)])

        # All scores should be zero
        self.assertTrue(all(res["score"] == 0 for res in results))

    def test_empty_query_list(self):
        """ Test behavior when query list is empty """
        with self.assertRaises(ValueError):
            get_bm25_similarities([], self.documents, self.ids)

    def test_empty_document_list(self):
        """ Test behavior when document list is empty """
        with self.assertRaises(ValueError):
            get_bm25_similarities(self.queries, [], [])

    def test_score_normalization(self):
        """ Test that scores are properly normalized """
        results = get_bm25_similarities(self.queries, self.documents, self.ids)
        max_score = max(res["score"] for res in results) if results else 0

        # Max score should be normalized to 1.0
        self.assertAlmostEqual(max_score, 1.0, places=2)


if __name__ == "__main__":
    unittest.main()
