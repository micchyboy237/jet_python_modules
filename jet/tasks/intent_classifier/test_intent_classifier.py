import pytest
from typing import Dict
from jet.tasks.intent_classifier import classify_intents
from jet.tasks.intent_classifier.utils import ClassificationResult, Id2Label, transform_label
from jet.models.model_registry.transformers.bert_model_registry import BERTModelRegistry
from transformers import PreTrainedTokenizer, AutoModelForSequenceClassification


class TestIntentClassifier:
    @pytest.fixture
    def model_registry(self):
        registry = BERTModelRegistry()
        yield registry
        registry.clear()

    @pytest.fixture
    def model_id(self):
        return "Falconsai/intent_classification"

    @pytest.fixture
    def texts(self):
        return [
            "I want to cancel my order",
            "Can you tell me about shipping options?",
            "I need to speak to a customer service representative",
        ]

    @pytest.fixture
    def id2label(self):
        return {
            "0": "cancellation",
            "1": "ordering",
            "2": "shipping",
            "3": "invoicing",
            "4": "billing and payment",
            "5": "returns and refunds",
            "6": "complaints and feedback",
            "7": "speak to person",
            "8": "edit account",
            "9": "delete account",
            "10": "delivery information",
            "11": "subscription",
            "12": "recover password",
            "13": "registration problems",
            "14": "appointment",
        }

    def test_classify_intents(self, model_registry, model_id, texts, id2label):
        # Arrange
        model = model_registry.load_model(
            model_id, {"device": "cpu", "precision": "fp32"})
        tokenizer = model_registry.get_tokenizer(model_id)
        expected = [
            ClassificationResult(
                label="cancellation",
                score=pytest.approx(0.9, abs=0.1),
                value=0,
                text="I want to cancel my order",
                doc_index=0,
                rank=1,
            ),
            ClassificationResult(
                label="shipping",
                score=pytest.approx(0.9, abs=0.1),
                value=2,
                text="Can you tell me about shipping options?",
                doc_index=1,
                rank=2,
            ),
            ClassificationResult(
                label="speak to person",
                score=pytest.approx(0.9, abs=0.1),
                value=7,
                text="I need to speak to a customer service representative",
                doc_index=2,
                rank=3,
            ),
        ]

        # Act
        results = classify_intents(
            model, tokenizer, texts, batch_size=2, show_progress=False)

        # Assert
        for result, expected_result in zip(results, expected):
            assert result["label"] == expected_result["label"]
            assert result["value"] == expected_result["value"]
            assert result["text"] == expected_result["text"]
            assert result["doc_index"] == expected_result["doc_index"]
            assert result["rank"] == expected_result["rank"]
            assert result["score"] == pytest.approx(
                expected_result["score"], abs=0.1)

    def test_transform_label_valid(self, id2label):
        # Arrange
        index = 0
        expected = "cancellation"

        # Act
        result = transform_label(index, id2label)

        # Assert
        assert result == expected

    def test_transform_label_invalid(self, id2label):
        # Arrange
        index = 999
        expected_error = f"Label index {index} is not found in id2label mapping"

        # Act & Assert
        with pytest.raises(IndexError, match=expected_error):
            transform_label(index, id2label)
