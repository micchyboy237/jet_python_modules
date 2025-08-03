import pytest
import numpy as np
from jet.vectors.mmr import get_diverse_texts, MMRResult


class TestGetDiverseTexts:
    @pytest.fixture
    def setup_data(self):
        # Sample embeddings (normalized for cosine similarity)
        query_embedding = np.array([[1.0, 0.0]])
        text_embeddings = np.array([
            [1.0, 0.0],   # Identical to query
            [0.0, 1.0],   # Orthogonal to query
            [0.707, 0.707]  # Diagonal
        ])
        texts = ["Text 1", "Text 2", "Text 3"]
        return query_embedding, text_embeddings, texts

    def test_basic_mmr_selection(self, setup_data):
        # Given: A query embedding, text embeddings, and texts
        query_embedding, text_embeddings, texts = setup_data
        mmr_lambda = 0.5
        num_results = 2

        # When: We call get_diverse_texts
        result = get_diverse_texts(
            query_embedding, text_embeddings, texts, mmr_lambda, num_results
        )

        # Then: Expect two results, first most relevant, second most diverse
        expected = [
            {"index": 0, "text": "Text 1", "similarity": pytest.approx(1.0)},
            {"index": 1, "text": "Text 2", "similarity": pytest.approx(0.0)}
        ]
        assert result == expected

    def test_with_initial_indices(self, setup_data):
        # Given: A query embedding, text embeddings, texts, and initial indices
        query_embedding, text_embeddings, texts = setup_data
        mmr_lambda = 0.5
        num_results = 2
        initial_indices = [2]  # Pre-select "Text 3"

        # When: We call get_diverse_texts with initial indices
        result = get_diverse_texts(
            query_embedding, text_embeddings, texts, mmr_lambda, num_results, initial_indices
        )

        # Then: Expect initial index included, followed by most diverse
        expected = [
            {"index": 2, "text": "Text 3",
                "similarity": pytest.approx(0.707, abs=1e-3)},
            {"index": 1, "text": "Text 2", "similarity": pytest.approx(0.0)}
        ]
        assert result == expected

    def test_invalid_inputs(self, setup_data):
        # Given: Invalid inputs
        query_embedding, text_embeddings, texts = setup_data

        # When/Then: Expect ValueError for empty query
        with pytest.raises(ValueError, match="Query embedding cannot be empty"):
            get_diverse_texts(np.array([]), text_embeddings, texts)

        # When/Then: Expect ValueError for mismatched embeddings and texts
        with pytest.raises(ValueError, match="Number of text embeddings must match number of texts"):
            get_diverse_texts(query_embedding, text_embeddings, ["Text 1"])

        # When/Then: Expect ValueError for invalid mmr_lambda
        with pytest.raises(ValueError, match="mmr_lambda must be between 0 and 1"):
            get_diverse_texts(query_embedding, text_embeddings,
                              texts, mmr_lambda=1.5)

        # When/Then: Expect ValueError for invalid initial indices
        with pytest.raises(ValueError, match="Initial indices out of range"):
            get_diverse_texts(query_embedding, text_embeddings,
                              texts, initial_indices=[3])

    def test_all_texts_selected(self, setup_data):
        # Given: A query embedding, text embeddings, and texts with no num_results limit
        query_embedding, text_embeddings, texts = setup_data
        mmr_lambda = 0.5

        # When: We call get_diverse_texts without num_results
        result = get_diverse_texts(
            query_embedding, text_embeddings, texts, mmr_lambda)

        # Then: Expect all texts returned in order of MMR score
        expected = [
            {"index": 0, "text": "Text 1", "similarity": pytest.approx(1.0)},
            {"index": 1, "text": "Text 2", "similarity": pytest.approx(0.0)},
            {"index": 2, "text": "Text 3",
                "similarity": pytest.approx(0.707, abs=1e-3)}
        ]
        assert result == expected

    def test_lambda_extremes(self, setup_data):
        # Given: A query embedding, text embeddings, and texts
        query_embedding, text_embeddings, texts = setup_data
        num_results = 2

        # When: We call get_diverse_texts with mmr_lambda = 1.0 (max relevance)
        result_relevance = get_diverse_texts(
            query_embedding, text_embeddings, texts, mmr_lambda=1.0, num_results=num_results
        )

        # Then: Expect results prioritizing relevance
        expected_relevance = [
            {"index": 0, "text": "Text 1", "similarity": pytest.approx(1.0)},
            {"index": 2, "text": "Text 3",
                "similarity": pytest.approx(0.707, abs=1e-3)}
        ]
        assert result_relevance == expected_relevance

        # When: We call get_diverse_texts with mmr_lambda = 0.0 (max diversity)
        result_diversity = get_diverse_texts(
            query_embedding, text_embeddings, texts, mmr_lambda=0.0, num_results=num_results
        )

        # Then: Expect results prioritizing diversity
        expected_diversity = [
            {"index": 0, "text": "Text 1", "similarity": pytest.approx(1.0)},
            {"index": 1, "text": "Text 2", "similarity": pytest.approx(0.0)}
        ]
        assert result_diversity == expected_diversity
