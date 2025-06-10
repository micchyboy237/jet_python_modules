# File: test_sentence_transformer_pooling.py
import pytest
import numpy as np
from jet.models.embeddings.sentence_transformer_pooling import encode_sentences, load_sentence_transformer


class TestSentenceTransformerPooling:
    @pytest.fixture
    def model_name(self):
        return "all-MiniLM-L6-v2"

    @pytest.fixture
    def sentences(self):
        return ["This is a test sentence.", "Another sentence for testing."]

    @pytest.fixture
    def pooling_modes(self):
        return ["cls_token", "mean_tokens", "max_tokens", "mean_sqrt_len_tokens"]

    def test_load_sentence_transformer(self, model_name, pooling_modes):
        """Test loading SentenceTransformer with different pooling modes."""
        expected_model_type = "sentence_transformers.SentenceTransformer"
        expected_embedding_dim = 384

        for mode in pooling_modes:
            result_model = load_sentence_transformer(
                model_name, pooling_mode=mode, use_mps=False)
            result_type = str(type(result_model))
            assert result_type.find(
                expected_model_type) != -1, f"Expected model type {expected_model_type} for {mode}"
            assert result_model[
                0].config.hidden_size == expected_embedding_dim, f"Expected embedding dim {expected_embedding_dim} for {mode}"

    def test_encode_sentences(self, model_name, sentences, pooling_modes):
        """Test encoding sentences with different pooling modes."""
        expected_shape = (len(sentences), 384)
        expected_embedding_type = np.ndarray

        for mode in pooling_modes:
            model = load_sentence_transformer(
                model_name, pooling_mode=mode, use_mps=False)
            result_embeddings = encode_sentences(
                model, sentences, batch_size=2)

            assert isinstance(
                result_embeddings, expected_embedding_type), f"Expected {expected_embedding_type} for {mode}"
            assert result_embeddings.shape == expected_shape, f"Expected shape {expected_shape} for {mode}"
            assert not np.any(np.isnan(result_embeddings)
                              ), f"Embeddings should not contain NaN for {mode}"
