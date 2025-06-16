
from typing import Optional, Self
from jet.models.model_registry.base import ModelFeatures
from jet.models.model_registry.transformers.mlx_model_registry import MLXModelRegistry
from jet.models.model_registry.transformers.bert_model_registry import BERTModelRegistry, ONNXBERTWrapper
from jet.models.model_registry.transformers.t5_model_registry import T5ModelRegistry, ONNXT5Wrapper
from jet.models.model_registry.transformers.llama_model_registry import LLaMAModelRegistry, ONNXLLaMAWrapper
from jet.models.model_registry.transformers.sentence_transformer_registry import SentenceTransformerRegistry, ONNXSentenceTransformerWrapper
from jet.models.model_registry.transformers.cross_encoder_model_registry import CrossEncoderModelRegistry, ONNXCrossEncoderWrapper
from jet.models.model_registry.non_transformers.xgboost_model_registry import XGBoostModelRegistry
from jet.models.model_registry.non_transformers.resnet_model_registry import ResNetModelRegistry
from jet.models.model_registry.non_transformers.random_forest_model_registry import RandomForestModelRegistry
from transformers import AutoModelForMaskedLM, T5ForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer, AutoConfig, PreTrainedTokenizer, PretrainedConfig
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.ensemble import RandomForestClassifier
import mlx.nn as nn
import onnxruntime as ort
import xgboost as xgb
import torch
import numpy as np
import os
import pickle
import pytest

# Centralized mock classes


class MockModel:
    def to(self, device: str) -> Self:
        return self

    def half(self) -> Self:
        return self


class MockCrossEncoderModel(MockModel):
    class InnerModel(MockModel):
        pass
    model = InnerModel()


class MockInferenceSession:
    def run(self, output_names: Optional[list], input_feed: dict) -> list:
        return [np.array([[[1]]])]

    def get_inputs(self) -> list[dict]:
        return [{"name": "input_ids"}, {"name": "attention_mask"}]


class MockTokenizer:
    def __call__(self, *args, **kwargs) -> dict:
        return {"input_ids": np.array([[1]]), "attention_mask": np.array([[1]])}


class MockConfig:
    pass


@pytest.fixture(autouse=True)
def setup():
    """Clear all registries before each test."""
    registry_classes = [
        MLXModelRegistry,
        BERTModelRegistry,
        T5ModelRegistry,
        LLaMAModelRegistry,
        SentenceTransformerRegistry,
        CrossEncoderModelRegistry,
        XGBoostModelRegistry,
        ResNetModelRegistry,
        RandomForestModelRegistry
    ]
    for registry_cls in registry_classes:
        registry = registry_cls()
        registry.clear()
    yield
    for registry_cls in registry_classes:
        registry = registry_cls()
        registry.clear()


class TestMLXModelRegistry:
    def test_load_model_cpu(self, monkeypatch):
        """Test loading an MLX model on CPU."""
        registry = MLXModelRegistry()
        model_id = "mlx-model"
        features: ModelFeatures = {"device": "cpu", "precision": "fp32"}

        def mock_load_mlx_model(self, model_id: str, features: ModelFeatures) -> nn.Module:
            assert features["device"] == "cpu", "Expected device to be cpu"
            return MockModel()

        monkeypatch.setattr(
            "jet.models.model_registry.transformers.mlx_model_registry.MLXModelRegistry._load_mlx_model",
            mock_load_mlx_model
        )

        result = registry.load_model(model_id, features)
        expected = MockModel
        assert isinstance(result, expected)
        assert model_id in registry._models

    def test_load_model_mps(self, monkeypatch):
        """Test loading an MLX model on MPS."""
        registry = MLXModelRegistry()
        model_id = "mlx-model"
        features: ModelFeatures = {"device": "mps", "precision": "fp32"}

        def mock_load_mlx_model(self, model_id: str, features: ModelFeatures) -> nn.Module:
            assert features["device"] == "mps", "Expected device to be mps"
            return MockModel()

        monkeypatch.setattr(
            "jet.models.model_registry.transformers.mlx_model_registry.MLXModelRegistry._load_mlx_model",
            mock_load_mlx_model
        )

        result = registry.load_model(model_id, features)
        expected = MockModel
        assert isinstance(result, expected)
        assert model_id in registry._models

    def test_load_model_cached(self, monkeypatch):
        """Test reusing a cached MLX model."""
        registry = MLXModelRegistry()
        model_id = "mlx-model"
        features: ModelFeatures = {"device": "mps", "precision": "fp32"}

        def mock_load_mlx_model(self, model_id: str, features: ModelFeatures) -> nn.Module:
            return MockModel()

        monkeypatch.setattr(
            "jet.models.model_registry.transformers.mlx_model_registry.MLXModelRegistry._load_mlx_model",
            mock_load_mlx_model
        )

        # Load model first time
        first_result = registry.load_model(model_id, features)
        # Load model second time (should be cached)
        second_result = registry.load_model(model_id, features)
        expected = MockModel
        assert isinstance(second_result, expected)
        assert first_result is second_result  # Ensure same instance (cached)
        assert model_id in registry._models

    def test_get_tokenizer(self, monkeypatch):
        """Test loading a tokenizer."""
        registry = MLXModelRegistry()
        model_id = "mlx-model"

        def mock_tokenizer(model_id: str) -> MockTokenizer:
            return MockTokenizer()

        monkeypatch.setattr(
            "transformers.AutoTokenizer.from_pretrained", mock_tokenizer)

        result = registry.get_tokenizer(model_id)
        expected = MockTokenizer
        assert isinstance(result, expected)
        assert model_id in registry._tokenizers

    def test_get_config(self, monkeypatch):
        """Test loading a config."""
        registry = MLXModelRegistry()
        model_id = "mlx-model"

        def mock_config(model_id: str) -> MockConfig:
            return MockConfig()

        monkeypatch.setattr(
            "transformers.AutoConfig.from_pretrained", mock_config)

        result = registry.get_config(model_id)
        expected = MockConfig
        assert isinstance(result, expected)
        assert model_id in registry._configs

    def test_invalid_model_id(self, monkeypatch):
        """Test loading with invalid model_id."""
        registry = MLXModelRegistry()
        model_id = "invalid-mlx-model"
        features: ModelFeatures = {"device": "cpu"}

        def mock_load_mlx_model(self, model_id: str, features: ModelFeatures) -> None:
            raise ValueError("Invalid model")

        monkeypatch.setattr(
            "jet.models.model_registry.transformers.mlx_model_registry.MLXModelRegistry._load_mlx_model",
            mock_load_mlx_model
        )

        with pytest.raises(ValueError) as exc_info:
            registry.load_model(model_id, features)
        expected_error = f"Could not load MLX model {model_id}"
        assert expected_error in str(exc_info.value)

    def test_clear(self, monkeypatch):
        """Test clearing the registry."""
        registry = MLXModelRegistry()
        model_id = "mlx-model"
        features: ModelFeatures = {"device": "mps", "precision": "fp32"}

        def mock_load_mlx_model(self, model_id: str, features: ModelFeatures) -> nn.Module:
            return MockModel()

        monkeypatch.setattr(
            "jet.models.model_registry.transformers.mlx_model_registry.MLXModelRegistry._load_mlx_model",
            mock_load_mlx_model
        )

        def mock_tokenizer(model_id: str) -> MockTokenizer:
            return MockTokenizer()

        monkeypatch.setattr(
            "transformers.AutoTokenizer.from_pretrained", mock_tokenizer)

        def mock_config(model_id: str) -> MockConfig:
            return MockConfig()

        monkeypatch.setattr(
            "transformers.AutoConfig.from_pretrained", mock_config)

        # Load a model, tokenizer, and config to populate the registry
        registry.load_model(model_id, features)
        registry.get_tokenizer(model_id)
        registry.get_config(model_id)
        assert model_id in registry._models
        assert model_id in registry._tokenizers
        assert model_id in registry._configs

        # Clear the registry
        registry.clear()
        expected_models = {}
        expected_tokenizers = {}
        expected_configs = {}
        assert registry._models == expected_models
        assert registry._tokenizers == expected_tokenizers
        assert registry._configs == expected_configs


class TestBERTModelRegistry:
    def test_load_model_pytorch(self, monkeypatch):
        """Test loading a PyTorch BERT model."""
        registry = BERTModelRegistry()
        model_id = "bert-base-uncased"
        features: ModelFeatures = {"device": "cpu", "precision": "fp32"}

        def mock_has_onnx(repo_id: str, token: Optional[str] = None) -> bool:
            return False

        monkeypatch.setattr(
            "jet.models.onnx_model_checker.has_onnx_model_in_repo", mock_has_onnx)

        def mock_load_model(model_id: str) -> MockModel:
            return MockModel()

        monkeypatch.setattr(
            "transformers.AutoModelForMaskedLM.from_pretrained", mock_load_model)

        result = registry.load_model(model_id, features)
        expected = MockModel
        assert isinstance(result, expected)
        assert model_id in registry._models

    def test_load_model_onnx(self, monkeypatch, tmp_path):
        """Test loading an ONNX BERT model."""
        registry = BERTModelRegistry()
        model_id = "bert-base-uncased"
        features: ModelFeatures = {"device": "cpu"}
        onnx_path = tmp_path / "model_arm64.onnx"

        def mock_has_onnx(repo_id: str, token: Optional[str] = None) -> bool:
            return repo_id == model_id

        def mock_get_onnx_paths(repo_id: str, cache_dir: Optional[str] = None) -> list[str]:
            return [str(onnx_path)] if repo_id == model_id else []

        monkeypatch.setattr(
            "jet.models.onnx_model_checker.has_onnx_model_in_repo", mock_has_onnx)
        monkeypatch.setattr(
            "jet.models.onnx_model_checker.get_onnx_model_paths", mock_get_onnx_paths)

        def mock_exists(path: str) -> bool:
            return path == str(onnx_path)

        monkeypatch.setattr(os.path, "exists", mock_exists)

        def mock_ort_session(path: str, providers: list[str] = None, sess_options: object = None) -> MockInferenceSession:
            return MockInferenceSession()

        monkeypatch.setattr(ort, "InferenceSession", mock_ort_session)

        def mock_tokenizer(model_id: str) -> MockTokenizer:
            return MockTokenizer()

        monkeypatch.setattr(
            "transformers.AutoTokenizer.from_pretrained", mock_tokenizer)

        result = registry.load_model(model_id, features)
        expected = ONNXBERTWrapper
        assert isinstance(result, expected)
        assert model_id in registry._models

    def test_get_tokenizer(self, monkeypatch):
        """Test loading a tokenizer."""
        registry = BERTModelRegistry()
        model_id = "bert-base-uncased"

        def mock_tokenizer(model_id: str) -> MockTokenizer:
            return MockTokenizer()

        monkeypatch.setattr(
            "transformers.AutoTokenizer.from_pretrained", mock_tokenizer)

        result = registry.get_tokenizer(model_id)
        expected = MockTokenizer
        assert isinstance(result, expected)
        assert model_id in registry._tokenizers

    def test_invalid_model_id(self, monkeypatch):
        """Test loading with invalid model_id."""
        registry = BERTModelRegistry()
        model_id = "invalid-model"
        features: ModelFeatures = {"device": "cpu"}

        def mock_has_onnx(repo_id: str, token: Optional[str] = None) -> bool:
            return False

        monkeypatch.setattr(
            "jet.models.onnx_model_checker.has_onnx_model_in_repo", mock_has_onnx)

        def mock_load_model(model_id: str) -> None:
            raise ValueError("Invalid model")

        monkeypatch.setattr(
            "transformers.AutoModelForMaskedLM.from_pretrained", mock_load_model)

        with pytest.raises(ValueError) as exc_info:
            registry.load_model(model_id, features)
        expected_error = f"Could not load BERT model {model_id}"
        assert expected_error in str(exc_info.value)


class TestT5ModelRegistry:
    def test_load_model_pytorch(self, monkeypatch):
        """Test loading a PyTorch T5 model."""
        registry = T5ModelRegistry()
        model_id = "t5-base"
        features: ModelFeatures = {"device": "cpu", "precision": "fp32"}

        def mock_has_onnx(repo_id: str, token: Optional[str] = None) -> bool:
            return False

        monkeypatch.setattr(
            "jet.models.onnx_model_checker.has_onnx_model_in_repo", mock_has_onnx)

        def mock_load_model(model_id: str) -> MockModel:
            return MockModel()

        monkeypatch.setattr(
            "transformers.T5ForConditionalGeneration.from_pretrained", mock_load_model)

        result = registry.load_model(model_id, features)
        expected = MockModel
        assert isinstance(result, expected)
        assert model_id in registry._models

    def test_load_model_onnx(self, monkeypatch, tmp_path):
        """Test loading an ONNX T5 model."""
        registry = T5ModelRegistry()
        model_id = "t5-base"
        features: ModelFeatures = {"device": "cpu"}
        onnx_path = tmp_path / "model_arm64.onnx"

        def mock_has_onnx(repo_id: str, token: Optional[str] = None) -> bool:
            return repo_id == model_id

        def mock_get_onnx_paths(repo_id: str, cache_dir: Optional[str] = None) -> list[str]:
            return [str(onnx_path)] if repo_id == model_id else []

        monkeypatch.setattr(
            "jet.models.onnx_model_checker.has_onnx_model_in_repo", mock_has_onnx)
        monkeypatch.setattr(
            "jet.models.onnx_model_checker.get_onnx_model_paths", mock_get_onnx_paths)

        def mock_exists(path: str) -> bool:
            return path == str(onnx_path)

        monkeypatch.setattr(os.path, "exists", mock_exists)

        def mock_ort_session(path: str, providers: list[str] = None, sess_options: object = None) -> MockInferenceSession:
            return MockInferenceSession()

        monkeypatch.setattr(ort, "InferenceSession", mock_ort_session)

        def mock_tokenizer(model_id: str) -> MockTokenizer:
            return MockTokenizer()

        monkeypatch.setattr(
            "transformers.AutoTokenizer.from_pretrained", mock_tokenizer)

        result = registry.load_model(model_id, features)
        expected = ONNXT5Wrapper
        assert isinstance(result, expected)
        assert model_id in registry._models

    def test_get_config(self, monkeypatch):
        """Test loading a config."""
        registry = T5ModelRegistry()
        model_id = "t5-base"

        def mock_config(model_id: str) -> MockConfig:
            return MockConfig()

        monkeypatch.setattr(
            "transformers.AutoConfig.from_pretrained", mock_config)

        result = registry.get_config(model_id)
        expected = MockConfig
        assert isinstance(result, expected)
        assert model_id in registry._configs


class TestLLaMAModelRegistry:
    def test_load_model_pytorch(self, monkeypatch):
        """Test loading a PyTorch LLaMA model."""
        registry = LLaMAModelRegistry()
        model_id = "meta-llama/Llama-3-8b"
        features: ModelFeatures = {"device": "cpu", "precision": "fp32"}

        def mock_has_onnx(repo_id: str, token: Optional[str] = None) -> bool:
            return False

        monkeypatch.setattr(
            "jet.models.onnx_model_checker.has_onnx_model_in_repo", mock_has_onnx)

        def mock_load_model(model_id: str) -> MockModel:
            return MockModel()

        monkeypatch.setattr(
            "transformers.AutoModelForCausalLM.from_pretrained", mock_load_model)

        result = registry.load_model(model_id, features)
        expected = MockModel
        assert isinstance(result, expected)
        assert model_id in registry._models

    def test_load_model_onnx(self, monkeypatch, tmp_path):
        """Test loading an ONNX LLaMA model."""
        registry = LLaMAModelRegistry()
        model_id = "meta-llama/Llama-3-8b"
        features: ModelFeatures = {"device": "cpu"}
        onnx_path = tmp_path / "model_arm64.onnx"

        def mock_has_onnx(repo_id: str, token: Optional[str] = None) -> bool:
            return repo_id == model_id

        def mock_get_onnx_paths(repo_id: str, cache_dir: Optional[str] = None) -> list[str]:
            return [str(onnx_path)] if repo_id == model_id else []

        monkeypatch.setattr(
            "jet.models.onnx_model_checker.has_onnx_model_in_repo", mock_has_onnx)
        monkeypatch.setattr(
            "jet.models.onnx_model_checker.get_onnx_model_paths", mock_get_onnx_paths)

        def mock_exists(path: str) -> bool:
            return path == str(onnx_path)

        monkeypatch.setattr(os.path, "exists", mock_exists)

        def mock_ort_session(path: str, providers: list[str] = None, sess_options: object = None) -> MockInferenceSession:
            return MockInferenceSession()

        monkeypatch.setattr(ort, "InferenceSession", mock_ort_session)

        def mock_tokenizer(model_id: str) -> MockTokenizer:
            return MockTokenizer()

        monkeypatch.setattr(
            "transformers.AutoTokenizer.from_pretrained", mock_tokenizer)

        result = registry.load_model(model_id, features)
        expected = ONNXLLaMAWrapper
        assert isinstance(result, expected)
        assert model_id in registry._models

    def test_get_tokenizer(self, monkeypatch):
        """Test loading a tokenizer."""
        registry = LLaMAModelRegistry()
        model_id = "meta-llama/Llama-3-8b"

        def mock_tokenizer(model_id: str) -> MockTokenizer:
            return MockTokenizer()

        monkeypatch.setattr(
            "transformers.AutoTokenizer.from_pretrained", mock_tokenizer)

        result = registry.get_tokenizer(model_id)
        expected = MockTokenizer
        assert isinstance(result, expected)
        assert model_id in registry._tokenizers


class TestCrossEncoderModelRegistry:
    def test_load_model_pytorch(self, monkeypatch):
        """Test loading a PyTorch CrossEncoder model."""
        registry = CrossEncoderModelRegistry()
        model_id = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        features: ModelFeatures = {"device": "cpu", "precision": "fp32"}

        def mock_has_onnx(repo_id: str, token: Optional[str] = None) -> bool:
            return False

        monkeypatch.setattr(
            "jet.models.onnx_model_checker.has_onnx_model_in_repo", mock_has_onnx)

        def mock_load_model(model_id: str) -> MockCrossEncoderModel:
            return MockCrossEncoderModel()

        monkeypatch.setattr(
            "sentence_transformers.CrossEncoder", mock_load_model)

        result = registry.load_model(model_id, features)
        expected = MockCrossEncoderModel
        assert isinstance(result, expected)
        assert model_id in registry._models

    def test_load_model_onnx(self, monkeypatch, tmp_path):
        """Test loading an ONNX CrossEncoder model."""
        registry = CrossEncoderModelRegistry()
        model_id = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        features: ModelFeatures = {"device": "cpu"}
        onnx_path = tmp_path / "model_arm64.onnx"

        def mock_has_onnx(repo_id: str, token: Optional[str] = None) -> bool:
            return repo_id == model_id

        def mock_get_onnx_paths(repo_id: str, cache_dir: Optional[str] = None) -> list[str]:
            return [str(onnx_path)] if repo_id == model_id else []

        monkeypatch.setattr(
            "jet.models.onnx_model_checker.has_onnx_model_in_repo", mock_has_onnx)
        monkeypatch.setattr(
            "jet.models.onnx_model_checker.get_onnx_model_paths", mock_get_onnx_paths)

        def mock_exists(path: str) -> bool:
            return path == str(onnx_path)

        monkeypatch.setattr(os.path, "exists", mock_exists)

        def mock_ort_session(path: str, providers: list[str] = None, sess_options: object = None) -> MockInferenceSession:
            return MockInferenceSession()

        monkeypatch.setattr(ort, "InferenceSession", mock_ort_session)

        def mock_tokenizer(model_id: str) -> MockTokenizer:
            return MockTokenizer()

        monkeypatch.setattr(
            "transformers.AutoTokenizer.from_pretrained", mock_tokenizer)

        result = registry.load_model(model_id, features)
        expected = ONNXCrossEncoderWrapper
        assert isinstance(result, expected)
        assert model_id in registry._models

    def test_get_config(self, monkeypatch):
        """Test loading a config."""
        registry = CrossEncoderModelRegistry()
        model_id = "cross-encoder/ms-marco-MiniLM-L-6-v2"

        def mock_config(model_id: str) -> MockConfig:
            return MockConfig()

        monkeypatch.setattr(
            "transformers.AutoConfig.from_pretrained", mock_config)

        result = registry.get_config(model_id)
        expected = MockConfig
        assert isinstance(result, expected)
        assert model_id in registry._configs


class TestSentenceTransformerRegistry:
    def test_load_model_pytorch(self, monkeypatch):
        """Test loading a PyTorch SentenceTransformer model."""
        registry = SentenceTransformerRegistry()
        model_id = "all-MiniLM-L6-v2"
        features: ModelFeatures = {"device": "cpu", "precision": "fp32"}

        def mock_has_onnx(repo_id: str, token: Optional[str] = None) -> bool:
            return False

        monkeypatch.setattr(
            "jet.models.onnx_model_checker.has_onnx_model_in_repo", mock_has_onnx)

        def mock_load_model(model_id: str) -> MockModel:
            return MockModel()

        monkeypatch.setattr(
            "sentence_transformers.SentenceTransformer", mock_load_model)

        result = registry.load_model(model_id, features)
        expected = MockModel
        assert isinstance(result, expected)
        assert model_id in registry._models

    def test_load_model_onnx(self, monkeypatch, tmp_path):
        """Test loading an ONNX SentenceTransformer model."""
        registry = SentenceTransformerRegistry()
        model_id = "all-MiniLM-L6-v2"
        features: ModelFeatures = {"device": "cpu"}
        onnx_path = tmp_path / "model_arm64.onnx"

        def mock_has_onnx(repo_id: str, token: Optional[str] = None) -> bool:
            return repo_id == model_id

        def mock_get_onnx_paths(repo_id: str, cache_dir: Optional[str] = None) -> list[str]:
            return [str(onnx_path)] if repo_id == model_id else []

        monkeypatch.setattr(
            "jet.models.onnx_model_checker.has_onnx_model_in_repo", mock_has_onnx)
        monkeypatch.setattr(
            "jet.models.onnx_model_checker.get_onnx_model_paths", mock_get_onnx_paths)

        def mock_exists(path: str) -> bool:
            return path == str(onnx_path)

        monkeypatch.setattr(os.path, "exists", mock_exists)

        def mock_ort_session(path: str, providers: list[str] = None, sess_options: object = None) -> MockInferenceSession:
            return MockInferenceSession()

        monkeypatch.setattr(ort, "InferenceSession", mock_ort_session)

        def mock_tokenizer(model_id: str) -> MockTokenizer:
            return MockTokenizer()

        monkeypatch.setattr(
            "transformers.AutoTokenizer.from_pretrained", mock_tokenizer)

        result = registry.load_model(model_id, features)
        expected = ONNXSentenceTransformerWrapper
        assert isinstance(result, expected)
        assert model_id in registry._models

    def test_get_tokenizer(self, monkeypatch):
        """Test loading a tokenizer."""
        registry = SentenceTransformerRegistry()
        model_id = "all-MiniLM-L6-v2"

        def mock_tokenizer(model_id: str) -> MockTokenizer:
            return MockTokenizer()

        monkeypatch.setattr(
            "transformers.AutoTokenizer.from_pretrained", mock_tokenizer)

        result = registry.get_tokenizer(model_id)
        expected = MockTokenizer
        assert isinstance(result, expected)
        assert model_id in registry._tokenizers

    def test_invalid_model_id(self, monkeypatch):
        """Test loading with invalid model_id."""
        registry = SentenceTransformerRegistry()
        model_id = "invalid-model"
        features: ModelFeatures = {"device": "cpu"}

        def mock_has_onnx(repo_id: str, token: Optional[str] = None) -> bool:
            return False

        monkeypatch.setattr(
            "jet.models.onnx_model_checker.has_onnx_model_in_repo", mock_has_onnx)

        def mock_load_model(model_id: str) -> None:
            raise ValueError("Invalid model")

        monkeypatch.setattr(
            "sentence_transformers.SentenceTransformer", mock_load_model)

        with pytest.raises(ValueError) as exc_info:
            registry.load_model(model_id, features)
        expected_error = f"Could not load SentenceTransformer model {model_id}"
        assert expected_error in str(exc_info.value)


class TestXGBoostModelRegistry:
    def test_load_model(self, monkeypatch, tmp_path):
        """Test loading an XGBoost model."""
        registry = XGBoostModelRegistry()
        model_id = str(tmp_path / "xgboost_model.pkl")

        class MockBooster:
            def predict(self, data: object) -> np.ndarray:
                return np.array([1])

        mock_model = MockBooster()
        with open(model_id, "wb") as f:
            pickle.dump(mock_model, f)

        result = registry.load_model(model_id)
        expected = MockBooster
        assert isinstance(result, expected)
        assert model_id in registry._models

    def test_get_tokenizer_raises(self):
        """Test that get_tokenizer raises ValueError."""
        registry = XGBoostModelRegistry()
        model_id = "xgboost_model.pkl"

        with pytest.raises(ValueError) as exc_info:
            registry.get_tokenizer(model_id)
        expected_error = f"Tokenizers are not applicable for non-transformer model {model_id}"
        assert expected_error in str(exc_info.value)

    def test_invalid_file_path(self, tmp_path):
        """Test loading with invalid file path."""
        registry = XGBoostModelRegistry()
        model_id = str(tmp_path / "nonexistent.pkl")

        with pytest.raises(ValueError) as exc_info:
            registry.load_model(model_id)
        expected_error = f"Could not load XGBoost model {model_id}"
        assert expected_error in str(exc_info.value)


class TestResNetModelRegistry:
    def test_load_model_pytorch(self, monkeypatch):
        """Test loading a PyTorch ResNet model."""
        registry = ResNetModelRegistry()
        model_id = "resnet50"
        features: ModelFeatures = {"device": "cpu", "precision": "fp32"}

        def mock_load_model(weights: object) -> MockModel:
            return MockModel()

        monkeypatch.setattr("torchvision.models.resnet50", mock_load_model)

        result = registry.load_model(model_id, features)
        expected = MockModel
        assert isinstance(result, expected)
        assert model_id in registry._models

    def test_invalid_model_id(self, monkeypatch):
        """Test loading with invalid model_id."""
        registry = ResNetModelRegistry()
        model_id = "invalid-resnet"
        features: ModelFeatures = {"device": "cpu"}

        def mock_load_model(weights: object) -> None:
            raise ValueError("Invalid model")

        monkeypatch.setattr("torchvision.models.resnet50", mock_load_model)

        with pytest.raises(ValueError) as exc_info:
            registry.load_model(model_id, features)
        expected_error = f"Unsupported ResNet model_id: {model_id}"
        assert expected_error in str(exc_info.value)


class TestRandomForestModelRegistry:
    def test_load_model(self, monkeypatch, tmp_path):
        """Test loading a Random Forest model."""
        registry = RandomForestModelRegistry()
        model_id = str(tmp_path / "rf_model.pkl")

        class MockRF:
            def predict(self, X: np.ndarray) -> np.ndarray:
                return np.array([1])

        mock_model = MockRF()
        with open(model_id, "wb") as f:
            pickle.dump(mock_model, f)

        result = registry.load_model(model_id)
        expected = MockRF
        assert isinstance(result, expected)
        assert model_id in registry._models

    def test_get_config_raises(self):
        """Test that get_config raises ValueError."""
        registry = RandomForestModelRegistry()
        model_id = "rf_model.pkl"

        with pytest.raises(ValueError) as exc_info:
            registry.get_config(model_id)
        expected_error = f"Configs are not applicable for non-transformer model {model_id}"
        assert expected_error in str(exc_info.value)

    def test_invalid_file_path(self, tmp_path):
        """Test loading with invalid file path."""
        registry = RandomForestModelRegistry()
        model_id = str(tmp_path / "nonexistent.pkl")

        with pytest.raises(ValueError) as exc_info:
            registry.load_model(model_id)
        expected_error = f"Could not load Random Forest model {model_id}"
        assert expected_error in str(exc_info.value)
