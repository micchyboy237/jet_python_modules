import pytest
from typing import Tuple, List
import numpy as np
from sentence_transformers import CrossEncoder

from jet.models.model_registry.transformers.cross_encoder_model_registry import CrossEncoderRegistry


@pytest.fixture
def registry():
    CrossEncoderRegistry.clear()
    return CrossEncoderRegistry()


class TestCrossEncoderRegistry:
    def test_load_model_success(self, registry):
        # Given: A valid CrossEncoder model ID
        model_id = "cross-encoder/ms-marco-MiniLM-L12-v2"
        max_length = 128

        # When: Loading the model
        model = registry.load_model(
            model_id=model_id, max_length=max_length, device="cpu")

        # Then: Model is loaded correctly
        expected = CrossEncoder
        result = type(model)
        assert result == expected, f"Expected model type {expected}, got {result}"
        assert registry.model_id == model_id
        assert registry.max_length == max_length

    def test_load_model_invalid_id(self, registry):
        # Given: An invalid model ID
        model_id = "invalid-model"

        # When: Attempting to load the model
        # Then: Raises ValueError
        with pytest.raises(ValueError):
            registry.load_model(model_id=model_id)

    def test_get_tokenizer(self, registry):
        # Given: A loaded model
        model_id = "cross-encoder/ms-marco-MiniLM-L12-v2"
        registry.load_model(model_id=model_id)

        # When: Retrieving the tokenizer
        tokenizer = registry.get_tokenizer()

        # Then: Tokenizer is returned
        expected = type(registry._load_tokenizer())
        result = type(tokenizer)
        assert result == expected, f"Expected tokenizer type {expected}, got {result}"

    def test_get_config(self, registry):
        # Given: A loaded model
        model_id = "cross-encoder/ms-marco-MiniLM-L12-v2"
        registry.load_model(model_id=model_id)

        # When: Retrieving the config
        config = registry.get_config()

        # Then: Config is returned with expected properties
        expected = "cross_encoder"
        result = config.model_type
        assert result == expected, f"Expected model_type {expected}, got {result}"

    def test_predict_scores_single_pair(self, registry):
        # Given: A loaded model and a single sentence pair
        model_id = "cross-encoder/ms-marco-MiniLM-L12-v2"
        registry.load_model(model_id=model_id)
        input_pair = ("Query: What is Python?",
                      "Python is a programming language.")

        # When: Predicting score
        scores = registry.predict_scores(input_pair, return_format="list")

        # Then: A single float score is returned
        expected = float
        result = type(scores)
        assert result == expected, f"Expected score type {expected}, got {result}"
        assert 0 <= scores <= 1, f"Score {scores} out of range [0,1]"

    def test_predict_scores_multiple_pairs(self, registry):
        # Given: A loaded model and multiple sentence pairs
        model_id = "cross-encoder/ms-marco-MiniLM-L12-v2"
        registry.load_model(model_id=model_id)
        input_pairs = [
            ("Query: What is Python?", "Python is a programming language."),
            ("Query: What is Java?", "Java is a coffee brand.")
        ]

        # When: Predicting scores
        scores = registry.predict_scores(
            input_pairs, batch_size=2, return_format="numpy")

        # Then: A numpy array of scores is returned
        expected = np.ndarray
        result = type(scores)
        assert result == expected, f"Expected scores type {expected}, got {result}"
        assert scores.shape == (2,), f"Expected shape (2,), got {scores.shape}"
        assert np.all((scores >= 0) & (scores <= 1)
                      ), f"Scores {scores} out of range [0,1]"

    def test_predict_scores_no_model(self, registry):
        # Given: No model loaded
        registry.model_id = None
        input_pair = ("Query: What is Python?",
                      "Python is a programming language.")

        # When: Attempting to predict scores
        # Then: Raises ValueError
        with pytest.raises(ValueError, match="No model_id set"):
            registry.predict_scores(input_pair)

    def test_clear_registry(self, registry):
        # Given: A loaded model and tokenizer
        model_id = "cross-encoder/ms-marco-MiniLM-L12-v2"
        registry.load_model(model_id=model_id)
        registry.get_tokenizer()
        registry.get_config()

        # When: Clearing the registry
        registry.clear()

        # Then: All caches are empty
        expected = 0
        result_models = len(registry._models)
        result_tokenizers = len(registry._tokenizers)
        result_configs = len(registry._configs)
        assert result_models == expected, f"Expected {expected} models, got {result_models}"
        assert result_tokenizers == expected, f"Expected {expected} tokenizers, got {result_tokenizers}"
        assert result_configs == expected, f"Expected {expected} configs, got {result_configs}"
