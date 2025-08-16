import pytest
import psutil
import torch
from jet.models.model_registry.transformers.sentence_transformer_registry import SentenceTransformerRegistry
from jet.models.model_types import EmbedModelType
from sentence_transformers import SentenceTransformer


@pytest.fixture
def clear_registry():
    """Fixture to clear the registry before and after each test."""
    SentenceTransformerRegistry.clear()
    yield
    SentenceTransformerRegistry.clear()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


class TestSentenceTransformerRegistry:
    """Tests for SentenceTransformerRegistry functionality and memory management."""

    def test_load_model_static_method(self, clear_registry):
        """Test that load_model static method works as expected."""
        # Given: A model ID and parameters
        model_id = "all-MiniLM-L6-v2"
        truncate_dim = 384
        expected_model_type = SentenceTransformer

        # When: Loading a model using the static method
        model = SentenceTransformerRegistry.load_model(
            model_id=model_id, truncate_dim=truncate_dim
        )

        # Then: The model is loaded correctly and is of the expected type
        assert isinstance(model, expected_model_type), (
            f"Expected model to be {expected_model_type}, got {type(model)}"
        )
        assert model.get_sentence_embedding_dimension() == truncate_dim, (
            f"Expected embedding dimension {truncate_dim}, got {model.get_sentence_embedding_dimension()}"
        )

    def test_model_caching(self, clear_registry):
        """Test that model caching reuses the same model instance."""
        # Given: A model ID and parameters
        model_id = "all-MiniLM-L6-v2"
        truncate_dim = 384

        # When: Loading the same model twice
        model1 = SentenceTransformerRegistry.load_model(
            model_id=model_id, truncate_dim=truncate_dim
        )
        model2 = SentenceTransformerRegistry.load_model(
            model_id=model_id, truncate_dim=truncate_dim
        )

        # Then: The same model instance is returned
        assert model1 is model2, "Expected cached model to be reused"

    def test_memory_management(self, clear_registry):
        """Test that memory usage does not grow uncontrollably with multiple model loads."""
        # Given: A clean registry and memory baseline
        process = psutil.Process()
        initial_mem = process.memory_info().rss / 1024 / 1024  # MB
        model_ids = [
            "all-MiniLM-L6-v2",
            "all-mpnet-base-v2",
            "all-distilroberta-v1",
            "paraphrase-MiniLM-L3-v2",
        ]

        # When: Loading multiple models beyond cache size
        for model_id in model_ids:
            SentenceTransformerRegistry.load_model(
                model_id=model_id, truncate_dim=384)

        # Then: Memory usage should not exceed a reasonable threshold
        final_mem = process.memory_info().rss / 1024 / 1024  # MB
        mem_increase = final_mem - initial_mem
        expected_max_increase = 2000  # MB, adjust based on expected model size
        assert mem_increase < expected_max_increase, (
            f"Memory increase ({mem_increase:.2f} MB) exceeds expected maximum ({expected_max_increase} MB)"
        )
