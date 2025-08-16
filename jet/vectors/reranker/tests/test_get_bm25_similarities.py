import pytest
from typing import List, Optional, Dict
from unittest.mock import patch
from collections import Counter
from bm25 import get_bm25_similarities, SimilarityResult

# Mocked dependencies


def mock_preprocess_texts(texts: List[str]) -> List[str]:
    return [text.lower() for text in texts]


def mock_get_words(text: str) -> List[str]:
    return text.split()


def mock_generate_unique_hash() -> str:
    return "mocked_hash"


class TestGetBM25Similarities:
    @pytest.fixture(autouse=True)
    def setup(self):
        # Patch dependencies for all tests
        self.patcher1 = patch("bm25.preprocess_texts",
                              side_effect=mock_preprocess_texts)
        self.patcher2 = patch("bm25.get_words", side_effect=mock_get_words)
        self.patcher3 = patch("bm25.generate_unique_hash",
                              side_effect=mock_generate_unique_hash)
        self.patcher1.start()
        self.patcher2.start()
        self.patcher3.start()
        yield
        # Cleanup
        self.patcher1.stop()
        self.patcher2.stop()
        self.patcher3.stop()

    def test_single_term_query_matching(self):
        # Given: A single-term query and documents with varying matches
        queries = ["apple"]
        documents = ["Apple pie is delicious",
                     "Banana bread is sweet", "Apple and orange juice"]
        ids = ["doc1", "doc2", "doc3"]
        metadatas = [{"source": "recipe1"}, {
            "source": "recipe2"}, {"source": "recipe3"}]
        expected = [
            {
                "rank": 1,
                "id": "doc1",
                "score": pytest.approx(0.833, rel=1e-2),
                "similarity": pytest.approx(0.833, rel=1e-2),
                "matched": {"apple": 1},
                "text": "Apple pie is delicious",
                "metadata": {"source": "recipe1"}
            },
            {
                "rank": 2,
                "id": "doc3",
                "score": pytest.approx(1.0, rel=1e-2),
                "similarity": pytest.approx(1.0, rel=1e-2),
                "matched": {"apple": 1},
                "text": "Apple and orange juice",
                "metadata": {"source": "recipe3"}
            },
            {
                "rank": 3,
                "id": "doc2",
                "score": 0.0,
                "similarity": 0.0,
                "matched": {},
                "text": "Banana bread is sweet",
                "metadata": {"source": "recipe2"}
            }
        ]

        # When: We compute BM25+ similarities
        result = get_bm25_similarities(
            queries, documents, ids=ids, metadatas=metadatas)

        # Then: The results should match expected rankings and scores
        for r, e in zip(result, expected):
            assert r["rank"] == e["rank"]
            assert r["id"] == e["id"]
            assert r["score"] == e["score"]
            assert r["similarity"] == e["similarity"]
            assert r["matched"] == e["matched"]
            assert r["text"] == e["text"]
            assert r["metadata"] == e["metadata"]

    def test_multi_term_query_ngram_boosting(self):
        # Given: Queries with different n-grams and documents with partial matches
        queries = ["apple pie", "apple", "pie crust"]
        documents = [
            "Apple pie is delicious",
            "Apple and orange juice",
            "Pie crust recipe"
        ]
        expected = [
            {
                "rank": 1,
                "id": "mocked_hash",
                "score": pytest.approx(1.0, rel=1e-2),
                "similarity": pytest.approx(1.0, rel=1e-2),
                "matched": {"apple pie": 1, "apple": 1},
                "text": "Apple pie is delicious",
                "metadata": None
            },
            {
                "rank": 2,
                "id": "mocked_hash",
                "score": pytest.approx(0.5, rel=1e-2),
                "similarity": pytest.approx(0.5, rel=1e-2),
                "matched": {"apple": 1},
                "text": "Apple and orange juice",
                "metadata": None
            },
            {
                "rank": 3,
                "id": "mocked_hash",
                "score": pytest.approx(0.5, rel=1e-2),
                "similarity": pytest.approx(0.5, rel=1e-2),
                "matched": {"pie crust": 1},
                "text": "Pie crust recipe",
                "metadata": None
            }
        ]

        # When: We compute BM25+ similarities with n-gram boosting
        result = get_bm25_similarities(queries, documents)

        # Then: Documents matching multi-term queries (e.g., "apple pie") rank higher
        for r, e in zip(result, expected):
            assert r["rank"] == e["rank"]
            assert r["score"] == e["score"]
            assert r["similarity"] == e["similarity"]
            assert r["matched"] == e["matched"]
            assert r["text"] == e["text"]
            assert r["metadata"] == e["metadata"]

    def test_empty_queries_raises_error(self):
        # Given: Empty queries list
        queries = []
        documents = ["Apple pie is delicious"]

        # When: We attempt to compute BM25+ similarities
        # Then: A ValueError should be raised
        with pytest.raises(ValueError, match="queries and documents must not be empty"):
            get_bm25_similarities(queries, documents)

    def test_empty_documents_raises_error(self):
        # Given: Empty documents list
        queries = ["apple"]
        documents = []

        # When: We attempt to compute BM25+ similarities
        # Then: A ValueError should be raised
        with pytest.raises(ValueError, match="queries and documents must not be empty"):
            get_bm25_similarities(queries, documents)

    def test_mismatched_ids_raises_error(self):
        # Given: Documents and ids with different lengths
        queries = ["apple"]
        documents = ["Apple pie", "Banana bread"]
        ids = ["doc1"]

        # When: We attempt to compute BM25+ similarities
        # Then: A ValueError should be raised
        with pytest.raises(ValueError, match="documents and ids must have the same lengths"):
            get_bm25_similarities(queries, documents, ids=ids)

    def test_mismatched_metadatas_raises_error(self):
        # Given: Documents and metadatas with different lengths
        queries = ["apple"]
        documents = ["Apple pie", "Banana bread"]
        metadatas = [{"source": "recipe1"}]

        # When: We attempt to compute BM25+ similarities
        # Then: A ValueError should be raised
        with pytest.raises(ValueError, match="documents and metadatas must have the same lengths"):
            get_bm25_similarities(queries, documents, metadatas=metadatas)
