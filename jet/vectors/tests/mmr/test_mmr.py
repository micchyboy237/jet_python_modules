import pytest
import numpy as np
from typing import List
from jet.vectors.mmr import MMRResult, get_diverse_results


class TestGetDiverseResults:
    def setup_method(self):
        # Normalize embeddings to ensure valid cosine similarities
        self.query_embedding = np.array([[1.0, 0.0]], dtype=np.float32)
        self.text_embeddings = np.array([
            [1.0, 0.0],   # Text 0: identical to query
            [0.7, 0.7],   # Text 1: moderately similar to query
            [0.0, 1.0],   # Text 2: dissimilar to query
            [0.8, 0.6]    # Text 3: similar to Text 1
        ], dtype=np.float32)
        # Normalize embeddings
        self.text_embeddings /= np.linalg.norm(
            self.text_embeddings, axis=1, keepdims=True)
        self.texts = [
            "Query-like text",
            "Moderate text",
            "Different text",
            "Similar to moderate text"
        ]

    def test_selects_relevant_and_diverse_texts(self):
        # Given: A query and texts with varying similarities
        query_embedding = self.query_embedding
        text_embeddings = self.text_embeddings
        texts = self.texts
        mmr_lambda = 0.5
        num_results = 3
        initial_indices = []
        expected = [
            {"index": 0, "text": "Query-like text",
                "similarity": pytest.approx(1.0)},
            {"index": 2, "text": "Different text",
                "similarity": pytest.approx(0.0)},
            {"index": 1, "text": "Moderate text",
                "similarity": pytest.approx(0.707, abs=0.001)}
        ]

        # When: We call get_diverse_results
        result = get_diverse_results(
            query_embedding, text_embeddings, texts, mmr_lambda, num_results, initial_indices)

        # Then: It selects the most relevant text first, then diverse texts
        assert result == expected, "Expected relevant and diverse texts in order"

    def test_empty_texts_returns_empty_list(self):
        # Given: Empty texts and embeddings
        query_embedding = np.array([[1.0, 0.0]], dtype=np.float32)
        text_embeddings = np.array([], dtype=np.float32).reshape(0, 2)
        texts: List[str] = []
        mmr_lambda = 0.5
        num_results = 3
        initial_indices = []
        expected: List[MMRResult] = []

        # When: We call get_diverse_results
        result = get_diverse_results(
            query_embedding, text_embeddings, texts, mmr_lambda, num_results, initial_indices)

        # Then: It returns an empty list
        assert result == expected, "Expected empty list for empty inputs"

    def test_single_text_returns_single_result(self):
        # Given: A single text and embedding
        query_embedding = np.array([[1.0, 0.0]], dtype=np.float32)
        text_embeddings = np.array([[0.7, 0.7]], dtype=np.float32)
        text_embeddings /= np.linalg.norm(text_embeddings,
                                          axis=1, keepdims=True)
        texts = ["Single text"]
        mmr_lambda = 0.5
        num_results = 3
        initial_indices = []
        expected = [
            {"index": 0, "text": "Single text",
                "similarity": pytest.approx(0.707, abs=0.001)}
        ]

        # When: We call get_diverse_results
        result = get_diverse_results(
            query_embedding, text_embeddings, texts, mmr_lambda, num_results, initial_indices)

        # Then: It returns only the single text
        assert result == expected, "Expected single text result"

    def test_high_lambda_favors_relevance(self):
        # Given: A query and texts, with high lambda favoring relevance
        query_embedding = self.query_embedding
        text_embeddings = self.text_embeddings
        texts = self.texts
        mmr_lambda = 0.9
        num_results = 3
        initial_indices = []
        expected = [
            {"index": 0, "text": "Query-like text",
                "similarity": pytest.approx(1.0)},
            {"index": 1, "text": "Moderate text",
                "similarity": pytest.approx(0.707, abs=0.001)},
            {"index": 3, "text": "Similar to moderate text",
                "similarity": pytest.approx(0.80, abs=0.001)}
        ]

        # When: We call get_diverse_results
        result = get_diverse_results(
            query_embedding, text_embeddings, texts, mmr_lambda, num_results, initial_indices)

        # Then: It prioritizes relevance over diversity
        assert result == expected, "Expected relevant texts prioritized with high lambda"

    def test_low_lambda_favors_diversity(self):
        # Given: A query and texts, with low lambda favoring diversity
        query_embedding = self.query_embedding
        text_embeddings = self.text_embeddings
        texts = self.texts
        mmr_lambda = 0.1
        num_results = 3
        initial_indices = []
        expected = [
            {"index": 0, "text": "Query-like text",
                "similarity": pytest.approx(1.0)},
            {"index": 2, "text": "Different text",
                "similarity": pytest.approx(0.0)},
            {"index": 1, "text": "Moderate text",
                "similarity": pytest.approx(0.707, abs=0.001)}
        ]

        # When: We call get_diverse_results
        result = get_diverse_results(
            query_embedding, text_embeddings, texts, mmr_lambda, num_results, initial_indices)

        # Then: It prioritizes diversity over relevance
        assert result == expected, "Expected diverse texts prioritized with low lambda"

    def test_invalid_lambda_raises_value_error(self):
        # Given: An invalid mmr_lambda value
        query_embedding = self.query_embedding
        text_embeddings = self.text_embeddings
        texts = self.texts
        mmr_lambda = 1.5
        num_results = 3
        initial_indices = []

        # When: We call get_diverse_results
        # Then: It raises a ValueError
        with pytest.raises(ValueError, match="mmr_lambda must be between 0 and 1"):
            get_diverse_results(query_embedding, text_embeddings,
                                texts, mmr_lambda, num_results, initial_indices)

    def test_initial_indices_are_included(self):
        # Given: A query and texts, with one pre-selected text
        query_embedding = self.query_embedding
        text_embeddings = self.text_embeddings
        texts = self.texts
        mmr_lambda = 0.5
        num_results = 3
        initial_indices = [1]  # Pre-select "Moderate text"
        expected = [
            {"index": 1, "text": "Moderate text",
                "similarity": pytest.approx(0.707, abs=0.001)},
            {"index": 0, "text": "Query-like text",
                "similarity": pytest.approx(1.0)},
            {"index": 2, "text": "Different text",
                "similarity": pytest.approx(0.0)}
        ]

        # When: We call get_diverse_results with initial_indices
        result = get_diverse_results(
            query_embedding, text_embeddings, texts, mmr_lambda, num_results, initial_indices)

        # Then: It includes the initial text and selects diverse, relevant texts
        assert result == expected, "Expected initial text included with diverse and relevant texts"

    def test_invalid_initial_indices_raises_value_error(self):
        # Given: Invalid initial_indices
        query_embedding = self.query_embedding
        text_embeddings = self.text_embeddings
        texts = self.texts
        mmr_lambda = 0.5
        num_results = 3
        initial_indices = [4]  # Out of range

        # When: We call get_diverse_results
        # Then: It raises a ValueError
        with pytest.raises(ValueError, match="initial_indices must contain valid indices within range"):
            get_diverse_results(query_embedding, text_embeddings,
                                texts, mmr_lambda, num_results, initial_indices)
