from typing import List, Dict, Union
import pytest
import os
from coreml_text_classifier import classify_text_coreml


class TestCoreMLTextClassifier:
    @pytest.fixture
    def model_path(self) -> str:
        return os.path.expanduser("~/.cache/huggingface/hub/models--Falconsai--intent_classification/snapshots/630d0d4668170a2a64d8d80b04d9844415bd4367/coreml/text-classification/float32_model.mlpackage")

    def test_single_text_classification(self, model_path: str):
        # Arrange
        text = "Book a flight to New York"
        expected = [{"label": str, "score": float}]

        # Act
        result = classify_text_coreml(text, model_path)

        # Assert
        assert isinstance(result, List), "Result should be a list"
        assert len(result) == 1, "Expected single result"
        assert isinstance(result[0]["label"], str), "Label should be a string"
        assert isinstance(result[0]["score"], float), "Score should be a float"
        assert 0 <= result[0]["score"] <= 1, "Score should be between 0 and 1"

    def test_batch_text_classification(self, model_path: str):
        # Arrange
        texts = ["Book a flight", "Cancel my reservation"]
        expected = [
            {"label": str, "score": float},
            {"label": str, "score": float}
        ]

        # Act
        result = classify_text_coreml(texts, model_path, batch_size=2)

        # Assert
        assert isinstance(result, List), "Result should be a list"
        assert len(result) == 2, "Expected two results"
        for i, res in enumerate(result):
            assert isinstance(
                res["label"], str), f"Label for text {i+1} should be a string"
            assert isinstance(
                res["score"], float), f"Score for text {i+1} should be a float"
            assert 0 <= res["score"] <= 1, f"Score for text {i+1} should be between 0 and 1"

    def test_invalid_model_path(self):
        # Arrange
        invalid_path = "/invalid/path/to/model.mlpackage"
        text = "Test text"

        # Act & Assert
        with pytest.raises(FileNotFoundError):
            classify_text_coreml(text, invalid_path)

    def test_empty_input(self, model_path: str):
        # Arrange
        text = []

        # Act & Assert
        with pytest.raises(ValueError, match="Input text cannot be empty"):
            classify_text_coreml(text, model_path)
