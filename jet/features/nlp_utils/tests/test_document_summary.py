import pytest
from typing import List, Dict
from jet.features.nlp_utils.document_summary import get_document_summary, DocumentResult, DocumentSummary


class TestDocumentSummary:
    def test_single_document_single_query(self):
        texts = ["The quick brown fox jumps over the lazy dog"]
        queries = ["quick"]
        expected_result: List[DocumentResult] = [{
            'id': 'doc_0',
            'rank': 1,
            'doc_index': 0,
            'score': pytest.approx(14.29, rel=0.1),
            'text': texts[0],
            'tokens': 7,
            'matched': {'quick': pytest.approx(14.29, rel=0.1)}
        }]
        expected_matched: Dict[str, int] = {
            'quick': pytest.approx(14.29, rel=0.1)}
        expected: DocumentSummary = {
            'results': expected_result,
            'matched': expected_matched
        }
        result = get_document_summary(
            texts, queries, min_count=1, as_score=True)
        assert result['results'] == expected['results']
        assert result['matched'] == expected['matched']

    def test_multiple_documents_multiple_queries(self):
        texts = [
            "The quick brown fox jumps",
            "The lazy dog sleeps"
        ]
        queries = ["quick", "lazy"]
        expected_result: List[DocumentResult] = [
            {
                'id': 'doc_0',
                'rank': 1,
                'doc_index': 0,
                'score': pytest.approx(20.0, rel=0.1),
                'text': texts[0],
                'tokens': 5,
                'matched': {'quick': pytest.approx(20.0, rel=0.1)}
            },
            {
                'id': 'doc_1',
                'rank': 2,
                'doc_index': 1,
                'score': pytest.approx(20.0, rel=0.1),
                'text': texts[1],
                'tokens': 4,
                'matched': {'lazy': pytest.approx(20.0, rel=0.1)}
            }
        ]
        expected_matched: Dict[str, int] = {
            'quick': pytest.approx(20.0, rel=0.1),
            'lazy': pytest.approx(20.0, rel=0.1)
        }
        expected: DocumentSummary = {
            'results': expected_result,
            'matched': expected_matched
        }
        result = get_document_summary(
            texts, queries, min_count=1, as_score=True)
        assert result['results'] == expected['results']
        assert result['matched'] == expected['matched']

    def test_no_matches(self):
        texts = ["The quick brown fox jumps"]
        queries = ["missing"]
        expected_result: List[DocumentResult] = [{
            'id': 'doc_0',
            'rank': 1,
            'doc_index': 0,
            'score': 0.0,
            'text': texts[0],
            'tokens': 5,
            'matched': {}
        }]
        expected_matched: Dict[str, int] = {}
        expected: DocumentSummary = {
            'results': expected_result,
            'matched': expected_matched
        }
        result = get_document_summary(
            texts, queries, min_count=1, as_score=True)
        assert result['results'] == expected['results']
        assert result['matched'] == expected['matched']
