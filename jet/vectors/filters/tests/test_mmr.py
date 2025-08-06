import pytest
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from jet.vectors.filters import select_mmr_texts, DiverseResult


class TestMMRSelection:
    """Test suite for select_mmr_texts function using BDD principles."""

    @pytest.fixture
    def sample_data(self) -> tuple[np.ndarray, List[str], np.ndarray]:
        """Fixture providing sample embeddings, texts, and query for reuse."""
        texts = [
            "Machine learning is a method of data analysis.",
            "Deep learning uses neural networks for tasks.",
            "Python is a popular language for ML development.",
            "Unsupervised learning finds patterns in data."
        ]
        query = "machine learning"
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(
            texts, convert_to_numpy=True, show_progress_bar=False)
        query_embedding = model.encode(
            [query], convert_to_numpy=True, show_progress_bar=False)[0]
        return embeddings, texts, query_embedding

    def test_selects_most_relevant_first(self, sample_data):
        """Test that the most relevant text is selected first."""
        # Given: Sample embeddings, texts, and query
        embeddings, texts, query_embedding = sample_data
        expected = {
            "index": 0,
            "text": "Machine learning is a method of data analysis.",
            # Adjusted for SentenceTransformer
            "score": pytest.approx(0.7, abs=0.1)
        }

        # When: Run MMR with max_texts=1
        result = select_mmr_texts(
            embeddings, texts, query_embedding, max_texts=1)

        # Then: Most relevant text is selected with correct score
        assert len(result) == 1
        assert result[0]["index"] == expected["index"]
        assert result[0]["text"] == expected["text"]
        assert result[0]["score"] == expected["score"]

    def test_balances_relevance_and_diversity(self, sample_data):
        """Test that MMR selects a diverse set of texts."""
        # Given: Sample embeddings, texts, and query
        embeddings, texts, query_embedding = sample_data
        expected = [
            {"index": 0, "text": "Machine learning is a method of data analysis."},
            {"index": 3, "text": "Unsupervised learning finds patterns in data."}
        ]

        # When: Run MMR with max_texts=2, lambda=0.5
        result = select_mmr_texts(
            embeddings, texts, query_embedding, lambda_param=0.5, max_texts=2)

        # Then: Two texts are selected, one relevant and one diverse
        assert len(result) == 2
        assert result[0]["index"] == expected[0]["index"]
        assert result[0]["text"] == expected[0]["text"]
        assert result[1]["index"] == expected[1]["index"]
        assert result[1]["text"] == expected[1]["text"]
        # First is more relevant
        assert result[0]["score"] > result[1]["score"]
        assert result[1]["score"] >= 0  # No negative scores

    def test_handles_empty_texts(self):
        """Test that empty text list returns empty result."""
        # Given: Empty texts and embeddings
        embeddings = np.array([])
        texts = []
        query_embedding = np.array([0.8, 0.4, 0.2])
        expected: List[DiverseResult] = []

        # When: Run MMR with empty inputs
        result = select_mmr_texts(embeddings, texts, query_embedding)

        # Then: Empty list is returned
        assert result == expected

    def test_raises_on_mismatched_embeddings(self):
        """Test that mismatched embeddings and texts raise an error."""
        # Given: Mismatched embeddings and texts
        embeddings = np.array([[0.8, 0.4, 0.1], [0.7, 0.5, 0.2]])
        texts = ["Text 1"]
        query_embedding = np.array([0.8, 0.4, 0.2])

        # When/Then: Expect ValueError
        with pytest.raises(ValueError, match="Number of texts must match number of embeddings"):
            select_mmr_texts(embeddings, texts, query_embedding)

    def test_raises_on_invalid_lambda(self):
        """Test that invalid lambda_param raises an error."""
        # Given: Valid embeddings and texts, invalid lambda
        embeddings = np.array([[0.8, 0.4, 0.1]])
        texts = ["Text 1"]
        query_embedding = np.array([0.8, 0.4, 0.1])

        # When/Then: Expect ValueError for lambda > 1
        with pytest.raises(ValueError, match="lambda_param must be between 0 and 1"):
            select_mmr_texts(embeddings, texts,
                             query_embedding, lambda_param=1.5)

        # When/Then: Expect ValueError for lambda < 0
        with pytest.raises(ValueError, match="lambda_param must be between 0 and 1"):
            select_mmr_texts(embeddings, texts,
                             query_embedding, lambda_param=-0.1)

    def test_raises_on_invalid_max_texts(self):
        """Test that invalid max_texts raises an error."""
        # Given: Valid embeddings and texts, invalid max_texts
        embeddings = np.array([[0.8, 0.4, 0.1]])
        texts = ["Text 1"]
        query_embedding = np.array([0.8, 0.4, 0.1])

        # When/Then: Expect ValueError
        with pytest.raises(ValueError, match="max_texts must be at least 1"):
            select_mmr_texts(embeddings, texts, query_embedding, max_texts=0)

    def test_handles_zero_embeddings(self):
        """Test handling of zero embeddings."""
        # Given: Zero embedding with valid text
        embeddings = np.array([[0.0, 0.0, 0.0]])
        texts = ["Zero embedding text"]
        query_embedding = np.array([0.8, 0.4, 0.2])
        expected = [{"index": 0, "text": "Zero embedding text", "score": 0.0}]

        # When: Run MMR
        result = select_mmr_texts(
            embeddings, texts, query_embedding, max_texts=1)

        # Then: Text is selected with zero score
        assert len(result) == 1
        assert result[0]["index"] == expected[0]["index"]
        assert result[0]["text"] == expected[0]["text"]
        assert result[0]["score"] == expected[0]["score"]

    def test_ids_are_preserved(self, sample_data):
        """Test that provided IDs are preserved in results."""
        # Given: Sample data with custom IDs
        embeddings, texts, query_embedding = sample_data
        custom_ids = ["custom_1", "custom_2", "custom_3", "custom_4"]
        expected_id = "custom_1"  # Most relevant text

        # When: Run MMR with custom IDs
        result = select_mmr_texts(
            embeddings, texts, query_embedding, ids=custom_ids, max_texts=1)

        # Then: Custom ID is preserved
        assert len(result) == 1
        assert result[0]["id"] == expected_id

    def test_embedding_dimensions(self, sample_data):
        """Test that embeddings have correct dimensions."""
        # Given: Sample embeddings, texts, and query
        embeddings, _, query_embedding = sample_data

        # When/Then: Verify dimensions
        assert embeddings.shape[1] == 384  # all-MiniLM-L6-v2 dimension
        assert query_embedding.shape[0] == 384

    def test_no_negative_scores(self, sample_data):
        # Given: Sample embeddings, texts, and query
        embeddings, texts, query_embedding = sample_data
        # When: Run MMR with max_texts=3
        result = select_mmr_texts(
            embeddings, texts, query_embedding, max_texts=3)
        # Then: All scores are non-negative
        assert all(
            r["score"] >= 0 for r in result), "MMR scores must be non-negative"
