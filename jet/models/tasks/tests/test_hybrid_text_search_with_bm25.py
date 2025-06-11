import pytest
import numpy as np
from typing import List, Dict
from jet.logger import logger
from jet.vectors.document_types import HeaderDocument
from jet.models.tasks.hybrid_text_search_with_bm25 import search_texts, highlight_text, filter_by_headers_and_facets, get_bm25_scores, get_original_document


class TestSearchTexts:
    @pytest.fixture
    def sample_docs(self) -> List[HeaderDocument]:
        return [
            HeaderDocument(
                id="doc1",
                text="Introduction\n# Getting Started\nInstall the package.",
                metadata={"category": "docs", "original_index": 0}
            ),
            HeaderDocument(
                id="doc2",
                text="Advanced\n# Configuration\nSet up advanced options.",
                metadata={"category": "advanced", "original_index": 1}
            )
        ]

    def test_faceted_search(self, sample_docs: List[HeaderDocument]):
        query = "getting started"
        facets = {"category": "docs"}
        expected = {
            "id": "doc1",
            "doc_index": 0,
            "rank": 1,
            "headers": ["# Getting Started"],
            "metadata": {"category": "docs", "original_index": 0}
        }
        result = search_texts(query, sample_docs,
                              facets=facets, top_k=1, rerank_top_k=1)
        logger.debug("Faceted search result: %s", result)
        assert len(result) == 1, f"Expected 1 result, got {len(result)}"
        assert result[0]["id"] == expected["id"], f"Expected id {expected['id']}, got {result[0]['id']}"
        assert result[0]["metadata"]["category"] == expected["metadata"]["category"]
        assert any(h in result[0]["headers"] for h in expected["headers"]
                   ), f"Expected headers {expected['headers']}, got {result[0]['headers']}"

    def test_highlighting(self, sample_docs: List[HeaderDocument]):
        query = "install"
        expected = "<mark>install</mark>"
        result = search_texts(query, sample_docs, top_k=1, rerank_top_k=1)
        logger.debug("Highlighting result: %s",
                     result[0]["highlighted_text"] if result else "No result")
        assert len(result) > 0, "Expected at least one result"
        assert result[0]["highlighted_text"] == expected, f"Expected '{expected}', got '{result[0]['highlighted_text']}'"

    def test_typo_tolerance(self, sample_docs: List[HeaderDocument]):
        query = "instal"  # Typo
        expected = ["doc1"]
        result = search_texts(query, sample_docs,
                              typo_tolerance=True, top_k=1, rerank_top_k=1)
        logger.debug("Typo tolerance result: %s", result)
        assert len(result) > 0, "Expected at least one result"
        assert result[0]["id"] in expected, f"Expected id in {expected}, got {result[0]['id']}"


class TestHighlightText:
    def test_highlight_single_term(self):
        text = "Install the package."
        query = "install"
        expected = "<mark>install</mark>"
        result = highlight_text(text, query)
        logger.debug("Highlight result: %s", result)
        assert result == expected, f"Expected '{expected}', got '{result}'"

    def test_highlight_empty_query(self):
        text = "Install the package."
        query = ""
        expected = ""
        result = highlight_text(text, query)
        logger.debug("Empty query highlight result: %s", result)
        assert result == expected, f"Expected '{expected}', got '{result}'"


class TestFilterByHeadersAndFacets:
    def test_facet_filtering(self):
        chunks = [
            {"text": "Test", "headers": [
                "# Test"], "doc_id": "1", "doc_index": 0, "metadata": {"category": "docs"}},
            {"text": "Other", "headers": [
                "# Other"], "doc_id": "2", "doc_index": 1, "metadata": {"category": "blog"}}
        ]
        query = "test"
        facets = {"category": "docs"}
        expected = [{"doc_id": "1"}]
        result = filter_by_headers_and_facets(chunks, query, facets)
        logger.debug("Facet filtering result: %s", result)
        assert len(result) == 1, f"Expected 1 chunk, got {len(result)}"
        assert result[0]["doc_id"] == expected[0][
            "doc_id"], f"Expected doc_id {expected[0]['doc_id']}, got {result[0]['doc_id']}"


class TestBM25Scores:
    def test_get_bm25_scores(self):
        chunk_texts = ["This is a test document", "Another document"]
        query = "test document"
        # Expect relevant document to have highest score
        expected_scores = [1.0, 0.0]
        # Expect all scores to be valid floats
        expected_validity = [True, True]
        result = get_bm25_scores(chunk_texts, query)
        result_validity = [isinstance(score, float) and not np.isnan(
            score) for score in result]
        logger.debug("BM25 scores for query '%s': %s, expected: %s",
                     query, result, expected_scores)
        logger.debug("Validity check: %s, expected: %s",
                     result_validity, expected_validity)
        assert result == expected_scores, f"Expected scores {expected_scores}, got {result}"
        assert result_validity == expected_validity, f"Expected validity {expected_validity}, got {result_validity}"

    def test_typo_tolerance(self):
        chunk_texts = ["install package", "configure options"]
        query = "instal"  # Typo
        # Expect relevant document to score higher
        expected_scores = [1.0, 0.0]
        result = get_bm25_scores(chunk_texts, query, typo_tolerance=True)
        logger.debug("BM25 scores for query '%s': %s, expected: %s",
                     query, result, expected_scores)
        assert result == expected_scores, f"Expected scores {expected_scores}, got {result}"

    def test_empty_query(self):
        chunk_texts = ["install package", "configure options"]
        query = ""
        expected_scores = [0.0, 0.0]
        result = get_bm25_scores(chunk_texts, query, typo_tolerance=True)
        logger.debug("BM25 scores for empty query: %s, expected: %s",
                     result, expected_scores)
        assert result == expected_scores, f"Expected scores {expected_scores}, got {result}"

    def test_empty_chunks(self):
        chunk_texts = []
        query = "install"
        expected_scores = []
        result = get_bm25_scores(chunk_texts, query, typo_tolerance=True)
        logger.debug("BM25 scores for empty chunks: %s, expected: %s",
                     result, expected_scores)
        assert result == expected_scores, f"Expected scores {expected_scores}, got {result}"

    def test_single_token_document(self):
        chunk_texts = ["install", "configure"]
        query = "instal"
        expected_scores = [1.0, 0.0]
        result = get_bm25_scores(chunk_texts, query, typo_tolerance=True)
        logger.debug(
            "BM25 scores for single-token documents: %s, expected: %s", result, expected_scores)
        assert result == expected_scores, f"Expected scores {expected_scores}, got {result}"

    def test_non_matching_query(self):
        chunk_texts = ["install package", "configure options"]
        query = "xyz nonmatching"  # Query with no matching terms
        expected_scores = [0.0, 0.0]  # Expect all zero scores
        result = get_bm25_scores(chunk_texts, query, typo_tolerance=True)
        logger.debug("BM25 scores for non-matching query '%s': %s, expected: %s",
                     query, result, expected_scores)
        assert result == expected_scores, f"Expected scores {expected_scores}, got {result}"
