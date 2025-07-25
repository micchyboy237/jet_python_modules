from typing import Dict, Optional, TypedDict, Literal, Union
from threading import Lock
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, PreTrainedTokenizer, PretrainedConfig
import onnxruntime as ort
import numpy as np
import logging
import os
import torch
from .base import TransformersModelRegistry, ModelFeatures
from jet.models.onnx_model_checker import has_onnx_model_in_repo, get_onnx_model_paths

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ONNXLLaMAWrapper:
    """Wrapper for ONNX LLaMA model to mimic generate method."""

    def __init__(self, session: ort.InferenceSession, tokenizer: PreTrainedTokenizer):
        self.session = session
        self.tokenizer = tokenizer

    def generate(self, input_ids: np.ndarray, max_length: int = 50, **kwargs) -> np.ndarray:
        """
        Generate text using the ONNX LLaMA model.

        Args:
            input_ids: Input token IDs.
            max_length: Maximum length of generated sequence.
            **kwargs: Additional generation parameters.

        Returns:
            np.ndarray: Generated token IDs.
        """
        logger.info(f"Generating text with ONNX LLaMA model")
        input_feed = {"input_ids": input_ids}
        outputs = self.session.run(None, input_feed)
        return outputs[0]  # Assume first output is generated tokens


class LLaMAModelRegistry(TransformersModelRegistry):
    """Thread-safe registry for LLaMA models."""

    _instance = None
    _lock = Lock()
    _models: Dict[str, Union[AutoModelForCausalLM, ONNXLLaMAWrapper]] = {}
    _tokenizers: Dict[str, PreTrainedTokenizer] = {}
    _configs: Dict[str, PretrainedConfig] = {}

    def load_model(self, model_id: str, features: Optional[ModelFeatures] = None) -> Optional[Union[AutoModelForCausalLM, ONNXLLaMAWrapper]]:
        """
        Load or retrieve a LLaMA model by model_id.

        Args:
            model_id: The identifier of the model (e.g., 'meta-llama/Llama-3-8b').
            features: Optional configuration (e.g., device, precision).

        Returns:
            Optional[Union[AutoModelForCausalLM, ONNXLLaMAWrapper]]: The loaded model instance.

        Raises:
            ValueError: If the model_id is invalid or loading fails.
        """
        features = features or {}
        with self._lock:
            if model_id in self._models:
                logger.info(
                    f"Reusing existing LLaMA model for model_id: {model_id}")
                return self._models[model_id]

            session = self._load_onnx_session(model_id, features)
            if session:
                try:
                    tokenizer = self.get_tokenizer(model_id)
                    model = ONNXLLaMAWrapper(session, tokenizer)
                    self._models[model_id] = model
                    return model
                except Exception as e:
                    logger.error(
                        f"Failed to load tokenizer for ONNX LLaMA model {model_id}: {str(e)}")

            logger.info(
                f"Loading PyTorch LLaMA model for model_id: {model_id}")
            return self._load_pytorch_model(model_id, features)

    def _load_pytorch_model(self, model_id: str, features: ModelFeatures) -> AutoModelForCausalLM:
        """Load a PyTorch LLaMA model."""
        try:
            model = AutoModelForCausalLM.from_pretrained(model_id)
            device = self._select_device(features)
            model = model.to(device)
            if features.get("precision") == "fp16" and device != "cpu":
                model = model.half()
            self._models[model_id] = model
            return model
        except Exception as e:
            logger.error(
                f"Failed to load PyTorch LLaMA model {model_id}: {str(e)}")
            raise ValueError(
                f"Could not load LLaMA model {model_id}: {str(e)}")

    def get_tokenizer(self, model_id: str) -> Optional[PreTrainedTokenizer]:
        """
        Load or retrieve a tokenizer for LLaMA models.

        Args:
            model_id: The identifier of the model.

        Returns:
            Optional[PreTrainedTokenizer]: The loaded tokenizer instance.

        Raises:
            ValueError: If the model_id is invalid or loading fails.
        """
        with self._lock:
            if model_id in self._tokenizers:
                logger.info(f"Reusing tokenizer for model_id: {model_id}")
                return self._tokenizers[model_id]

            logger.info(f"Loading tokenizer for model_id: {model_id}")
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                self._tokenizers[model_id] = tokenizer
                return tokenizer
            except Exception as e:
                logger.error(f"Failed to load tokenizer {model_id}: {str(e)}")
                raise ValueError(
                    f"Could not load tokenizer {model_id}: {str(e)}")

    def get_config(self, model_id: str) -> Optional[PretrainedConfig]:
        """
        Load or retrieve a config for LLaMA models.

        Args:
            model_id: The identifier of the model.

        Returns:
            Optional[PretrainedConfig]: The loaded config instance.

        Raises:
            ValueError: If the model_id is invalid or loading fails.
        """
        with self._lock:
            if model_id in self._configs:
                logger.info(f"Reusing config for model_id: {model_id}")
                return self._configs[model_id]

            logger.info(f"Loading config for model_id: {model_id}")
            try:
                config = AutoConfig.from_pretrained(model_id)
                self._configs[model_id] = config
                return config
            except Exception as e:
                logger.error(f"Failed to load config {model_id}: {str(e)}")
                raise ValueError(f"Could not load config {model_id}: {str(e)}")

    def clear(self) -> None:
        """Clear all cached models, tokenizers, and configs."""
        with self._lock:
            self._models.clear()
            self._tokenizers.clear()
            self._configs.clear()
            self._onnx_sessions.clear()
            logger.info("LLaMA registry cleared")
