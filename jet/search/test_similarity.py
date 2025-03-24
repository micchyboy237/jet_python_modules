import unittest
from collections import defaultdict
from jet.search.similarity import get_bm25_similarities, adjust_score_with_rewards_and_penalties


class TestBM25SimilarityRanking(unittest.TestCase):

    def setUp(self):
        """Set up sample queries and documents with diverse lengths."""
        self.queries = [
            "apple",
            "banana smoothie",
            "cherry tart",
            "fresh orange juice",
            "grape",
            "blueberry pancakes",
            "healthy breakfast options"
        ]
        self.documents = [
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

    def test_top_results_are_diverse(self):
        """Ensure the top results immediately match multiple different queries."""
        results = get_bm25_similarities(self.queries, self.documents)

        # Check that the top results contain at least two unique query matches
        top_result = results[0]
        second_result = results[1] if len(results) > 1 else None

        self.assertGreater(
            len(top_result["matched"]), 1, "Top result should match multiple queries.")

        if second_result:
            self.assertGreater(len(
                second_result["matched"]), 0, "Second result should have at least one query match.")

        # Ensure diversity by checking distinct matches
        unique_queries_covered = set()
        for result in results[:3]:  # Check top 3 results
            unique_queries_covered.update(result["matched"].keys())

        self.assertGreaterEqual(len(unique_queries_covered), 3,
                                "Top 3 results should match at least 3 unique queries.")

    def test_results_are_sorted_correctly(self):
        """Ensure results are sorted by the adjusted score."""
        results = get_bm25_similarities(self.queries, self.documents)
        scores = [entry["score"] for entry in results]

        self.assertTrue(all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1)),
                        "Results should be sorted by descending score.")

    def test_results_contain_expected_queries(self):
        """Ensure results contain expected query matches."""
        results = get_bm25_similarities(self.queries, self.documents)

        # Check that every top-ranked result contains at least one query match
        for result in results[:3]:  # Check top 3 results
            self.assertTrue(any(q in result["matched"] for q in self.queries),
                            f"Result '{result['text']}' should match at least one query.")

    def test_low_relevance_documents_rank_lower(self):
        """Ensure documents with no relevant query matches rank at the bottom."""
        extra_docs = self.documents + \
            ["This document is completely unrelated."]
        results = get_bm25_similarities(self.queries, extra_docs)

        # Last result should have no matches
        self.assertEqual(len(results[-1]["matched"]), 0,
                         "Unrelated document should have zero query matches.")
        self.assertAlmostEqual(
            results[-1]["score"], 0, "Unrelated document should have the lowest score.")


if __name__ == "__main__":
    unittest.main()
