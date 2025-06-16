from typing import Dict, Optional, TypedDict, Literal, Union
from threading import Lock
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer, AutoConfig, PreTrainedTokenizer, PretrainedConfig
import onnxruntime as ort
import numpy as np
import logging
import os
import torch
from .base import TransformersModelRegistry, ModelFeatures
from jet.models.onnx_model_checker import has_onnx_model_in_repo, get_onnx_model_paths

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ONNXCrossEncoderWrapper:
    """Wrapper for ONNX CrossEncoder model to mimic predict method."""

    def __init__(self, session: ort.InferenceSession, tokenizer: PreTrainedTokenizer):
        self.session = session
        self.tokenizer = tokenizer

    def predict(self, sentence_pairs, batch_size=32, **kwargs) -> np.ndarray:
        """
        Predict reranking scores for sentence pairs using the ONNX model.

        Args:
            sentence_pairs: List of tuples (query, passage) to score.
            batch_size: Batch size for prediction.
            **kwargs: Additional tokenizer arguments.

        Returns:
            np.ndarray: Scores for each sentence pair.
        """
        logger.info(
            f"Predicting scores for {len(sentence_pairs)} sentence pairs with ONNX CrossEncoder model")
        all_scores = []
        for i in range(0, len(sentence_pairs), batch_size):
            batch = sentence_pairs[i:i + batch_size]
            inputs = self.tokenizer(
                [pair[0] for pair in batch],
                [pair[1] for pair in batch],
                return_tensors="np",
                padding=True,
                truncation=True,
                **kwargs
            )
            input_feed = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"]
            }
            if "token_type_ids" in [inp.name for inp in self.session.get_inputs()]:
                input_feed["token_type_ids"] = inputs.get(
                    "token_type_ids", np.zeros_like(inputs["input_ids"]))

            outputs = self.session.run(None, input_feed)
            scores = outputs[0]
            all_scores.append(scores)
        return np.concatenate(all_scores, axis=0)


class CrossEncoderModelRegistry(TransformersModelRegistry):
    """Thread-safe registry for CrossEncoder models."""

    _instance = None
    _lock = Lock()
    _models: Dict[str, Union[CrossEncoder, ONNXCrossEncoderWrapper]] = {}
    _tokenizers: Dict[str, PreTrainedTokenizer] = {}
    _configs: Dict[str, PretrainedConfig] = {}

    def load_model(self, model_id: str, features: Optional[ModelFeatures] = None) -> Optional[Union[CrossEncoder, ONNXCrossEncoderWrapper]]:
        """
        Load or retrieve a CrossEncoder model by model_id.

        Args:
            model_id: The identifier of the model (e.g., 'cross-encoder/ms-marco-MiniLM-L-6-v2').
            features: Optional configuration (e.g., device, precision).

        Returns:
            Optional[Union[CrossEncoder, ONNXCrossEncoderWrapper]]: The loaded model instance.

        Raises:
            ValueError: If the model_id is invalid or loading fails.
        """
        features = features or {}
        with self._lock:
            if model_id in self._models:
                logger.info(
                    f"Reusing existing CrossEncoder model for model_id: {model_id}")
                return self._models[model_id]

            session = self._load_onnx_session(model_id, features)
            if session:
                try:
                    tokenizer = self.get_tokenizer(model_id)
                    model = ONNXCrossEncoderWrapper(session, tokenizer)
                    self._models[model_id] = model
                    return model
                except Exception as e:
                    logger.error(
                        f"Failed to load tokenizer for ONNX CrossEncoder model {model_id}: {str(e)}")

            logger.info(
                f"Loading PyTorch CrossEncoder model for model_id: {model_id}")
            return self._load_pytorch_model(model_id, features)

    def _load_pytorch_model(self, model_id: str, features: ModelFeatures) -> CrossEncoder:
        """Load a PyTorch CrossEncoder model."""
        try:
            model = CrossEncoder(model_id)
            device = self._select_device(features)
            model.model = model.model.to(device)
            if features.get("precision") == "fp16" and device != "cpu":
                model.model = model.model.half()
            self._models[model_id] = model
            return model
        except Exception as e:
            logger.error(
                f"Failed to load PyTorch CrossEncoder model {model_id}: {str(e)}")
            raise ValueError(
                f"Could not load CrossEncoder model {model_id}: {str(e)}")

    def get_tokenizer(self, model_id: str) -> Optional[PreTrainedTokenizer]:
        """
        Load or retrieve a tokenizer for CrossEncoder models.

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
        Load or retrieve a config for CrossEncoder models.

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
            logger.info("CrossEncoder registry cleared")
