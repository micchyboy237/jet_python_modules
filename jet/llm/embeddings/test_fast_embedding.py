import pytest
import numpy as np
from jet.llm.embeddings.fast_embedding import generate_embeddings


class TestGenerateEmbeddings:
    def test_basic_embedding_generation(self):
        # Test basic functionality with small document set
        documents = ["This is a test document.",
                     "Another document for testing."]
        # Assuming all-MiniLM-L6-v2 with 384-dim embeddings
        expected_shape = (2, 384)
        expected_first_embedding_start = np.array(
            # First few values (approximate)
            [0.013, 0.024, -0.011], dtype=np.float32)

        result = generate_embeddings(documents, batch_size=2, use_mps=False)

        assert result.shape == expected_shape, f"Expected shape {expected_shape}, got {result.shape}"
        assert np.allclose(result[0, :3], expected_first_embedding_start, atol=1e-2), \
            f"Expected first embedding start {expected_first_embedding_start}, got {result[0, :3]}"

    def test_empty_document_list(self):
        # Test handling of empty input
        documents = []
        expected = np.array([])

        with pytest.raises(ValueError):
            result = generate_embeddings(documents)

    def test_single_document(self):
        # Test with single document
        documents = ["Single document test."]
        expected_shape = (1, 384)

        result = generate_embeddings(documents, batch_size=1, use_mps=False)

        assert result.shape == expected_shape, f"Expected shape {expected_shape}, got {result.shape}"
        assert isinstance(result, np.ndarray), "Result should be a numpy array"

    def test_large_batch(self):
        # Test with larger batch
        documents = ["Test document " + str(i) for i in range(100)]
        expected_shape = (100, 384)

        result = generate_embeddings(documents, batch_size=32, use_mps=False)

        assert result.shape == expected_shape, f"Expected shape {expected_shape}, got {result.shape}"
        assert not np.any(
            np.isnan(result)), "Embeddings should not contain NaN values"
