from typing import Dict, Literal, Union, List
import pytest
import torch
from jet.models.model_registry.transformers.bert_model_registry import BERTModelRegistry, ONNXBERTWrapper
from transformers import AutoModelForSequenceClassification, PreTrainedTokenizer
from JetScripts.models.model_registry.run_bert_intent_classifier import classify_intents


@pytest.fixture
def registry():
    """Fixture to provide a BERTModelRegistry instance."""
    reg = BERTModelRegistry()
    yield reg
    reg.clear()


@pytest.fixture
def model_id():
    """Fixture for the model ID."""
    return "Falconsai/intent_classification"


@pytest.fixture
def features():
    """Fixture for model features."""
    return {
        "device": "mps" if torch.backends.mps.is_available() else "cpu",
        "precision": "fp16",
    }


class TestBERTIntentClassifier:
    """Test class for BERT intent classification functionality."""

    def test_load_model_and_classify(self, registry, model_id, features):
        """Test loading the model and classifying intents."""
        # Arrange
        expected_results = [
            {"label": "cancellation", "score": pytest.approx(0.9, abs=0.1)},
            {"label": "shipping", "score": pytest.approx(0.9, abs=0.1)},
            {"label": "speak to person", "score": pytest.approx(0.9, abs=0.1)},
        ]
        texts = [
            "I want to cancel my order",
            "Can you tell me about shipping options?",
            "I need to speak to a customer service representative",
        ]

        # Act
        model = registry.load_model(model_id, features)
        tokenizer = registry.get_tokenizer(model_id)
        assert model is not None, "Model failed to load"
        assert tokenizer is not None, "Tokenizer failed to load"
        result = classify_intents(model, tokenizer, texts, batch_size=32)

        # Assert
        for res, exp in zip(result, expected_results):
            assert res["label"] == exp["label"], f"Expected label {exp['label']}, got {res['label']}"
            assert res["score"] == exp["score"], f"Expected score {exp['score']}, got {res['score']}"

    def test_model_reuse(self, registry, model_id, features):
        """Test that the registry reuses the same model instance."""
        # Arrange
        expected_model = registry.load_model(model_id, features)
        assert expected_model is not None, "First model load failed"

        # Act
        result_model = registry.load_model(model_id, features)

        # Assert
        assert result_model is expected_model, "Registry did not reuse the same model instance"

    def test_clear_registry(self, registry, model_id, features):
        """Test clearing the registry."""
        # Arrange
        registry.load_model(model_id, features)
        registry.get_tokenizer(model_id)
        registry.get_config(model_id)

        # Act
        registry.clear()
        result_models = registry._models
        result_tokenizers = registry._tokenizers
        result_configs = registry._configs
        result_onnx_sessions = registry._onnx_sessions

        # Assert
        expected_empty: Dict = {}
        assert result_models == expected_empty, "Models not cleared"
        assert result_tokenizers == expected_empty, "Tokenizers not cleared"
        assert result_configs == expected_empty, "Configs not cleared"
        assert result_onnx_sessions == expected_empty, "ONNX sessions not cleared"
