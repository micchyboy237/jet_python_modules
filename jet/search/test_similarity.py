import pytest
from jet.vectors.reranker.bm25 import get_bm25_similarities


class TestGetBM25Similarities:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.queries = [
            "machine learning in healthcare",
            "deep learning for image recognition",
            "AI in financial technology"
        ]
        self.documents = [
            "Deep learning methods are widely used in image recognition tasks.",
            "Machine learning has numerous applications in healthcare.",
            "AI is transforming financial services with automation and analytics.",
            "A general introduction to computer science.",
        ]

    def test_top_results_are_diverse(self):
        """Ensure the top results immediately match multiple different queries."""
        results = get_bm25_similarities(self.queries, self.documents)

        top_result = results[0]
        second_result = results[1] if len(results) > 1 else None

        assert len(top_result["matched"]
                   ) > 1, "Top result should match multiple queries."

        if second_result:
            assert len(
                second_result["matched"]) > 0, "Second result should have at least one query match."

        unique_queries_covered = set()
        for result in results[:3]:
            unique_queries_covered.update(result["matched"].keys())

        assert len(
            unique_queries_covered) >= 3, "Top 3 results should match at least 3 unique queries."

    def test_results_are_sorted_correctly(self):
        """Ensure results are sorted by the adjusted score."""
        results = get_bm25_similarities(self.queries, self.documents)
        scores = [entry["score"] for entry in results]

        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1)), \
            "Results should be sorted by descending score."

    def test_results_contain_expected_queries(self):
        """Ensure results contain expected query matches."""
        results = get_bm25_similarities(self.queries, self.documents)

        for result in results[:3]:
            assert any(q in result["matched"] for q in self.queries), \
                f"Result '{result['text']}' should match at least one query."

    def test_low_relevance_documents_rank_lower(self):
        """Ensure documents with no relevant query matches rank at the bottom."""
        extra_docs = self.documents + \
            ["This document is completely unrelated."]
        results = get_bm25_similarities(self.queries, extra_docs)

        assert len(results[-1]["matched"]) == 0, \
            "Unrelated document should have zero query matches."
        assert pytest.approx(results[-1]["score"], 0.01) == 0, \
            "Unrelated document should have the lowest score."
