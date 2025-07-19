import pytest
import numpy as np
from typing import List, Dict, Union
from unittest.mock import patch
from jet.vectors.semantic_search.base import vector_search, cosine_similarity
from jet.vectors.semantic_search.search_types import Match, SearchResult
from jet.models.model_types import EmbedModelType


class MockEmbedModel:
    pass


class TestVectorSearch:
    @patch("jet.vectors.semantic_search.base.preprocess_texts")
    @patch("jet.vectors.semantic_search.base.generate_embeddings")
    @patch("jet.vectors.semantic_search.base.get_words")
    def test_longer_ngram_ranks_higher(self, mock_get_words, mock_generate_embeddings, mock_preprocess_texts):
        """Test that a document with a longer n-gram match ranks higher than one with multiple shorter matches."""
        # Given a query and two documents with different match patterns
        query = "react native development"
        texts = [
            "React Native development experience required.",
            "React skills needed. Native app experience preferred."
        ]
        embed_model = MockEmbedModel()
        mock_preprocess_texts.return_value = texts
        mock_get_words.return_value = ["react", "native", "development"]
        query_emb = np.array([1.0, 0.0])
        text_emb1 = np.array(
            [0.8 / np.linalg.norm([0.8, 0.2]), 0.2 / np.linalg.norm([0.8, 0.2])])
        text_emb2 = np.array(
            [0.85 / np.linalg.norm([0.85, 0.15]), 0.15 / np.linalg.norm([0.85, 0.15])])
        mock_generate_embeddings.return_value = np.array(
            [query_emb, query_emb, text_emb1, text_emb2])

        # When vector search is performed
        expected_results = [
            SearchResult(
                rank=1,
                score=pytest.approx(
                    1.0 * (1 + 1.5 * (np.log1p(24) / np.log1p(100))), 0.01),
                header="React Native development experience required.",
                content="",
                id="id1",
                metadata={},
                matches=[Match(text="react native development",
                               start_idx=0, end_idx=24)]
            ),
            SearchResult(
                rank=2,
                score=pytest.approx(
                    0.9701425001453319 * (1 + 1.5 * (np.log1p(6) / np.log1p(100))), 0.01),
                header="React skills needed. Native app experience preferred.",
                content="",
                id="id2",
                metadata={},
                matches=[
                    Match(text="react", start_idx=0, end_idx=5),
                    Match(text="native", start_idx=21, end_idx=27)
                ]
            )
        ]
        results = vector_search(
            query=query,
            texts=texts,
            embed_model=embed_model,
            top_k=2,
            ids=["id1", "id2"],
            metadatas=[{}, {}]
        )

        # Then the document with the longer n-gram match ranks higher
        assert len(results) == 2, "Should return two results"
        assert results[0]["rank"] == 1, "First result should have rank 1"
        assert results[1]["rank"] == 2, "Second result should have rank 2"
        assert results[0]["id"] == "id1", "First result should be the exact match document"
        assert results[1]["id"] == "id2", "Second result should be the partial match document"
        assert results[0]["score"] > results[1]["score"], "Exact match should have higher score"
        assert pytest.approx(results[0]["score"],
                             0.01) == expected_results[0]["score"]
        assert pytest.approx(results[1]["score"],
                             0.01) == expected_results[1]["score"]
        assert results[0]["matches"] == expected_results[0]["matches"], "Matches for first doc should be correct"
        assert results[1]["matches"] == expected_results[1]["matches"], "Matches for second doc should be correct"

    @patch("jet.vectors.semantic_search.base.preprocess_texts")
    @patch("jet.vectors.semantic_search.base.generate_embeddings")
    @patch("jet.vectors.semantic_search.base.get_words")
    def test_empty_query_raises_error(self, mock_get_words, mock_generate_embeddings, mock_preprocess_texts):
        """Test that an empty query raises a ValueError."""
        # Given
        query: List[str] = []
        texts = ["React Native development experience required."]
        embed_model = MockEmbedModel()

        # When
        with pytest.raises(ValueError) as exc_info:
            vector_search(query=query, texts=texts, embed_model=embed_model)

        # Then
        assert str(exc_info.value) == "Query list cannot be empty"

    @patch("jet.vectors.semantic_search.base.preprocess_texts")
    @patch("jet.vectors.semantic_search.base.generate_embeddings")
    @patch("jet.vectors.semantic_search.base.get_words")
    def test_mismatched_ids_raises_error(self, mock_get_words, mock_generate_embeddings, mock_preprocess_texts):
        """Test that mismatched ids and texts raise a ValueError."""
        # Given
        query = "react native"
        texts = ["React Native development experience required."]
        embed_model = MockEmbedModel()
        ids = ["id1", "id2"]  # More IDs than texts
        mock_preprocess_texts.return_value = texts
        mock_generate_embeddings.return_value = np.array(
            [[1.0, 0.0], [0.9, 0.1]])
        mock_get_words.return_value = ["react", "native"]

        # When
        with pytest.raises(ValueError) as exc_info:
            vector_search(query=query, texts=texts,
                          embed_model=embed_model, ids=ids)

        # Then
        assert str(exc_info.value) == "Length of ids must match length of texts"

    @patch("jet.vectors.semantic_search.base.preprocess_texts")
    @patch("jet.vectors.semantic_search.base.generate_embeddings")
    @patch("jet.vectors.semantic_search.base.get_words")
    def test_no_matches_still_ranks_by_similarity(self, mock_get_words, mock_generate_embeddings, mock_preprocess_texts):
        """Test that documents with no matches are ranked by cosine similarity."""
        # Given a query and two documents, one with a match and one without
        query = "python"
        texts = [
            "React Native development experience required.",
            "Python programming skills needed."
        ]
        embed_model = MockEmbedModel()
        mock_preprocess_texts.return_value = texts
        mock_get_words.return_value = ["python"]
        query_emb = np.array([1.0, 0.0])
        text_emb1 = np.array(
            [0.7 / np.linalg.norm([0.7, 0.3]), 0.3 / np.linalg.norm([0.7, 0.3])])
        text_emb2 = np.array(
            [0.8 / np.linalg.norm([0.8, 0.2]), 0.2 / np.linalg.norm([0.8, 0.2])])
        mock_generate_embeddings.return_value = np.array(
            [query_emb, query_emb, text_emb1, text_emb2])

        # When vector search is performed
        expected_results = [
            SearchResult(
                rank=1,
                score=pytest.approx(
                    0.9191450300180579 * (1 + 1.5 * (np.log1p(6) / np.log1p(100))), 0.01),
                header="Python programming skills needed.",
                content="",
                id="id2",
                metadata={},
                matches=[Match(text="python", start_idx=0, end_idx=6)]
            ),
            SearchResult(
                rank=2,
                score=pytest.approx(1.0, 0.01),
                header="React Native development experience required.",
                content="",
                id="id1",
                metadata={},
                matches=[]
            )
        ]
        results = vector_search(
            query=query,
            texts=texts,
            embed_model=embed_model,
            top_k=2,
            ids=["id1", "id2"],
            metadatas=[{}, {}]
        )

        # Then the document with a match ranks first
        assert len(results) == 2, "Should return two results"
        assert results[0]["id"] == "id2", "Document with match should rank first"
        assert results[1]["id"] == "id1", "Document without match should rank second"
        assert results[0]["score"] > results[1]["score"], "Matched document should have higher score"
        assert pytest.approx(results[0]["score"],
                             0.01) == expected_results[0]["score"]
        assert pytest.approx(results[1]["score"],
                             0.01) == expected_results[1]["score"]
        assert results[0]["matches"] == expected_results[0]["matches"], "Matches for first doc should be correct"
        assert results[1]["matches"] == expected_results[1]["matches"], "Matches for second doc should be correct"
