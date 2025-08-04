import pytest
import numpy as np
from typing import List
from jet.vectors.filters import select_mmr_texts, DiverseResult


class TestSelectMMRTexts:
    @pytest.fixture
    def sample_data(self):
        """Provide sample embeddings and texts for testing."""
        # Simulated embeddings for 5 texts (3D for simplicity)
        embeddings = np.array([
            [1.0, 0.0, 0.0],  # Text 0: Highly relevant to query
            [0.9, 0.1, 0.0],  # Text 1: Similar to Text 0
            [0.0, 1.0, 0.0],  # Text 2: Different topic
            [0.1, 0.9, 0.0],  # Text 3: Similar to Text 2
            [0.0, 0.0, 1.0]   # Text 4: Unique topic
        ])
        texts = [
            "AI improves healthcare diagnostics",
            "Machine learning enhances medical imaging",
            "Renewable energy powers smart grids",
            "Solar panels reduce carbon emissions",
            "Quantum computing advances cryptography"
        ]
        query_embedding = np.array([1.0, 0.0, 0.0])  # Query similar to Text 0
        ids = ["id0", "id1", "id2", "id3", "id4"]
        return embeddings, texts, query_embedding, ids

    def test_empty_input(self):
        # Given: Empty embeddings and texts
        embeddings = np.array([])
        texts = []
        query_embedding = np.array([1.0, 0.0])
        # When: Call select_mmr_texts
        result = select_mmr_texts(embeddings, texts, query_embedding)
        # Then: Expect empty list
        expected: List[DiverseResult] = []
        assert result == expected, f"Expected {expected}, got {result}"

    def test_single_text(self, sample_data):
        # Given: Single text and embedding
        embeddings = sample_data[0][:1]
        texts = sample_data[1][:1]
        query_embedding = sample_data[2]
        ids = sample_data[3][:1]
        # When: Call select_mmr_texts
        result = select_mmr_texts(embeddings, texts, query_embedding, ids=ids)
        # Then: Expect single result with correct fields
        expected: List[DiverseResult] = [{
            "id": "id0",
            "index": 0,
            "text": "AI improves healthcare diagnostics",
            "score": pytest.approx(1.0, abs=1e-5)
        }]
        assert result == expected, f"Expected {expected}, got {result}"

    def test_diverse_selection_lambda_0_5(self, sample_data):
        # Given: Sample data with query favoring healthcare
        embeddings, texts, query_embedding, ids = sample_data
        # When: Call select_mmr_texts with lambda=0.5, max_texts=3
        result = select_mmr_texts(
            embeddings, texts, query_embedding, lambda_param=0.5, max_texts=3, ids=ids)
        # Then: Expect diverse texts (healthcare, quantum, and renewable energy topic)
        expected_texts = [
            "AI improves healthcare diagnostics",  # Most relevant
            "Quantum computing advances cryptography",  # Diverse
        ]
        expected_renewable_texts = [
            "Renewable energy powers smart grids",  # Renewable energy topic
            "Solar panels reduce carbon emissions"  # Renewable energy topic
        ]
        result_texts = [r["text"] for r in result]
        assert len(result) == 3, f"Expected 3 results, got {len(result)}"
        assert result_texts[0] == expected_texts[
            0], f"Expected first text {expected_texts[0]}, got {result_texts[0]}"
        assert result_texts[1] == expected_texts[
            1], f"Expected second text {expected_texts[1]}, got {result_texts[1]}"
        assert result_texts[
            2] in expected_renewable_texts, f"Expected renewable energy text {expected_renewable_texts}, got {result_texts[2]}"
        assert result[0][
            "index"] == 0, f"First result should be index 0, got {result[0]['index']}"

    def test_high_lambda_favors_relevance(self, sample_data):
        # Given: Sample data with query favoring healthcare
        embeddings, texts, query_embedding, ids = sample_data
        # When: Call select_mmr_texts with lambda=0.9 (favor relevance), max_texts=2
        result = select_mmr_texts(
            embeddings, texts, query_embedding, lambda_param=0.9, max_texts=2, ids=ids)
        # Then: Expect healthcare-related texts (most relevant)
        expected_texts = [
            "AI improves healthcare diagnostics",  # Most relevant
            "Machine learning enhances medical imaging"  # Similar but less diverse
        ]
        result_texts = [r["text"] for r in result]
        assert len(result) == 2, f"Expected 2 results, got {len(result)}"
        assert result_texts == expected_texts, f"Expected {expected_texts}, got {result_texts}"

    def test_low_lambda_favors_diversity(self, sample_data):
        # Given: Sample data with query favoring healthcare
        embeddings, texts, query_embedding, ids = sample_data
        # When: Call select_mmr_texts with lambda=0.1 (favor diversity), max_texts=3
        result = select_mmr_texts(
            embeddings, texts, query_embedding, lambda_param=0.1, max_texts=3, ids=ids)
        # Then: Expect highly diverse texts
        expected_texts = [
            "AI improves healthcare diagnostics",  # Most relevant
            "Quantum computing advances cryptography",  # Diverse
            "Renewable energy powers smart grids"  # Diverse
        ]
        result_texts = [r["text"] for r in result]
        assert len(result) == 3, f"Expected 3 results, got {len(result)}"
        assert all(
            t in result_texts for t in expected_texts), f"Expected {expected_texts}, got {result_texts}"

    def test_invalid_lambda(self):
        # Given: Invalid lambda_param
        embeddings = np.array([[1.0, 0.0]])
        texts = ["Test text"]
        query_embedding = np.array([1.0, 0.0])
        # When: Call select_mmr_texts with lambda_param > 1
        with pytest.raises(ValueError) as exc_info:
            select_mmr_texts(embeddings, texts,
                             query_embedding, lambda_param=1.5)
        # Then: Expect ValueError
        assert str(exc_info.value) == "lambda_param must be between 0 and 1"

    def test_invalid_max_texts(self):
        # Given: Invalid max_texts
        embeddings = np.array([[1.0, 0.0]])
        texts = ["Test text"]
        query_embedding = np.array([1.0, 0.0])
        # When: Call select_mmr_texts with max_texts < 1
        with pytest.raises(ValueError) as exc_info:
            select_mmr_texts(embeddings, texts, query_embedding, max_texts=0)
        # Then: Expect ValueError
        assert str(exc_info.value) == "max_texts must be at least 1"

    def test_mismatched_query_dimension(self):
        # Given: Mismatched query embedding dimension
        embeddings = np.array([[1.0, 0.0, 0.0]])
        texts = ["Test text"]
        query_embedding = np.array([1.0, 0.0])  # Wrong dimension
        # When: Call select_mmr_texts
        with pytest.raises(ValueError) as exc_info:
            select_mmr_texts(embeddings, texts, query_embedding)
        # Then: Expect ValueError
        assert str(
            exc_info.value) == "Query embedding dimension must match text embeddings"
