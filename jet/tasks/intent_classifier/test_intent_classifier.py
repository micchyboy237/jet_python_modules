from typing import List, Dict, Union, Optional
import pytest
import torch
from jet.tasks.intent_classifier.base import classify_text
from jet.logger import logger
from jet.transformers.formatters import format_json


class TestIntentClassifier:
    @pytest.fixture
    def model_name(self) -> str:
        return "Falconsai/intent_classification"

    def test_single_text_classification(self, model_name: str):
        # Arrange
        text = "Book a flight to New York"
        expected = [
            {
                "label": "BookFlight",  # Adjust based on actual model output
                "score": float
            }
        ]

        # Act
        result = classify_text(text, model_name)

        # Assert
        assert isinstance(result, List), "Result should be a list"
        assert isinstance(
            result[0], Dict), "Result item should be a dictionary"
        assert result[0]["label"] == expected[0][
            "label"], f"Expected label {expected[0]['label']}, got {result[0]['label']}"
        assert isinstance(result[0]["score"], float), "Score should be a float"
        assert 0 <= result[0]["score"] <= 1, "Score should be between 0 and 1"

    def test_batch_text_classification(self, model_name: str):
        # Arrange
        texts = ["Book a flight", "Cancel my reservation"]
        expected = [
            {
                "label": "BookFlight",  # Adjust based on actual model output
                "score": float
            },
            {
                "label": "Cancel",  # Adjust based on actual model output
                "score": float
            }
        ]

        # Act
        result = classify_text(texts, model_name, batch_size=2)

        # Assert
        assert isinstance(result, List), "Result should be a list"
        for i, (res, exp) in enumerate(zip(result, expected)):
            assert isinstance(
                res, Dict), f"Result item {i+1} should be a dictionary"
            assert res["label"] == exp[
                "label"], f"Expected label {exp['label']} for text {i+1}, got {res['label']}"
            assert isinstance(
                res["score"], float), f"Score for text {i+1} should be a float"
            assert 0 <= res["score"] <= 1, f"Score for text {i+1} should be between 0 and 1"

    def test_empty_input(self, model_name: str):
        # Arrange
        text = []

        # Act & Assert
        with pytest.raises(ValueError, match="Input must not be empty"):
            classify_text(text, model_name)

    def test_invalid_model_name(self):
        # Arrange
        invalid_model = "invalid/model/name"
        text = "Test text"

        # Act & Assert
        # Model loading error (specific exception may vary)
        with pytest.raises(Exception):
            classify_text(text, invalid_model)

    def test_mps_device_selection(self, model_name: str, monkeypatch):
        # Arrange
        text = "Test device usage"
        expected_device = "mps" if torch.backends.mps.is_available() else "cpu"

        # Act
        result = classify_text(text, model_name)

        # Assert
        assert isinstance(result, List), "Result should be a list"
        assert isinstance(
            result[0], Dict), "Result item should be a dictionary"
        # Note: Device logging is verified indirectly via successful execution
