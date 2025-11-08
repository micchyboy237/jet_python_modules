import pytest
from typing import List
from jet.wordnet.histogram import TextAnalysis


@pytest.fixture
def sample_text_data() -> List[str]:
    return [
        "Artificial intelligence and machine learning are closely related fields.",
        "Machine learning enables systems to learn from data.",
        "Deep learning is a subfield of machine learning.",
        "Artificial intelligence applications are vast and growing rapidly.",
        "Natural language processing is a key domain of AI."
    ]


class TestFilterTopDocumentsByTfidfAndCollocations:
    # --- Given ---
    @pytest.fixture
    def text_analysis(self, sample_text_data):
        return TextAnalysis(sample_text_data)

    # --- When ---
    def test_returns_top_documents_sorted_by_combined_score(self, text_analysis: TextAnalysis, sample_text_data: List[str]):
        # When
        result = text_analysis.filter_top_documents_by_tfidf_and_collocations(
            ngram_range=(1, 2),
            weight_tfidf=0.6,
            weight_collocation=0.4,
            top_n=3
        )

        # Then
        expected = {
            "length": 3,
            "required_keys": {"doc_idx", "doc", "score_tfidf", "score_collocation", "score_combined"},
            "sorted": True,
            "original_docs": sample_text_data,
        }

        # --- Assertions ---
        assert isinstance(result, list)
        assert len(result) == expected["length"]

        for item in result:
            # Verify structure
            assert expected["required_keys"].issubset(item.keys())

            # Verify doc_idx is valid and doc matches original
            assert isinstance(item["doc_idx"], int)
            assert 0 <= item["doc_idx"] < len(expected["original_docs"])
            assert item["doc"] == expected["original_docs"][item["doc_idx"]]

        # Verify sorted order of combined scores
        combined_scores = [r["score_combined"] for r in result]
        expected_sorted = sorted(combined_scores, reverse=True)
        assert combined_scores == expected_sorted

    def test_combined_score_changes_with_weights(self, text_analysis: TextAnalysis):
        # When
        result_tfidf_heavy = text_analysis.filter_top_documents_by_tfidf_and_collocations(
            ngram_range=(1, 2),
            weight_tfidf=0.9,
            weight_collocation=0.1,
            top_n=3
        )
        result_colloc_heavy = text_analysis.filter_top_documents_by_tfidf_and_collocations(
            ngram_range=(1, 2),
            weight_tfidf=0.1,
            weight_collocation=0.9,
            top_n=3
        )

        # Then
        expected = {
            "different_scores": True
        }

        score_pairs = zip(result_tfidf_heavy, result_colloc_heavy)
        differences = [
            abs(a["score_combined"] - b["score_combined"]) for a, b in score_pairs
        ]

        # --- Assertion ---
        assert any(diff > 0.0001 for diff in differences) == expected["different_scores"]

    def test_handles_empty_data_gracefully(self):
        # Given
        empty_analysis = TextAnalysis([])

        # When
        result = empty_analysis.filter_top_documents_by_tfidf_and_collocations()

        # Then
        expected = []  # no documents to rank
        assert result == expected
