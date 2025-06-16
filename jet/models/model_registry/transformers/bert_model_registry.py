from typing import Dict, Optional, TypedDict, Literal, Union
from threading import Lock
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoConfig, PreTrainedTokenizer, PretrainedConfig
import onnxruntime as ort
import numpy as np
import logging
import os
import torch
from .base import TransformersModelRegistry, ModelFeatures
from jet.models.onnx_model_checker import has_onnx_model_in_repo, get_onnx_model_paths

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ONNXBERTWrapper:
    """Wrapper for ONNX BERT model to provide embeddings."""

    def __init__(self, session: ort.InferenceSession, tokenizer: PreTrainedTokenizer):
        self.session = session
        self.tokenizer = tokenizer

    def get_embeddings(self, texts, batch_size=32, **kwargs) -> np.ndarray:
        """
        Get embeddings for texts using the ONNX BERT model.

        Args:
            texts: List of strings to encode.
            batch_size: Batch size for encoding.
            **kwargs: Additional tokenizer arguments.

        Returns:
            np.ndarray: Embeddings for the input texts.
        """
        logger.info(f"Encoding {len(texts)} texts with ONNX BERT model")
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch,
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
            embeddings = outputs[0].mean(axis=1)  # Pool last hidden state
            all_embeddings.append(embeddings)
        return np.concatenate(all_embeddings, axis=0)


class BERTModelRegistry(TransformersModelRegistry):
    """Thread-safe registry for BERT models."""

    _instance = None
    _lock = Lock()
    _models: Dict[str, Union[AutoModelForMaskedLM, ONNXBERTWrapper]] = {}
    _tokenizers: Dict[str, PreTrainedTokenizer] = {}
    _configs: Dict[str, PretrainedConfig] = {}

    def load_model(self, model_id: str, features: Optional[ModelFeatures] = None) -> Optional[Union[AutoModelForMaskedLM, ONNXBERTWrapper]]:
        """
        Load or retrieve a BERT model by model_id.

        Args:
            model_id: The identifier of the model (e.g., 'bert-base-uncased').
            features: Optional configuration (e.g., device, precision).

        Returns:
            Optional[Union[AutoModelForMaskedLM, ONNXBERTWrapper]]: The loaded model instance.

        Raises:
            ValueError: If the model_id is invalid or loading fails.
        """
        features = features or {}
        with self._lock:
            if model_id in self._models:
                logger.info(
                    f"Reusing existing BERT model for model_id: {model_id}")
                return self._models[model_id]

            session = self._load_onnx_session(model_id, features)
            if session:
                try:
                    tokenizer = self.get_tokenizer(model_id)
                    model = ONNXBERTWrapper(session, tokenizer)
                    self._models[model_id] = model
                    return model
                except Exception as e:
                    logger.error(
                        f"Failed to load tokenizer for ONNX BERT model {model_id}: {str(e)}")

            logger.info(f"Loading PyTorch BERT model for model_id: {model_id}")
            return self._load_pytorch_model(model_id, features)

    def _load_pytorch_model(self, model_id: str, features: ModelFeatures) -> AutoModelForMaskedLM:
        """Load a PyTorch BERT model."""
        try:
            model = AutoModelForMaskedLM.from_pretrained(model_id)
            device = self._select_device(features)
            model = model.to(device)
            if features.get("precision") == "fp16" and device != "cpu":
                model = model.half()
            self._models[model_id] = model
            return model
        except Exception as e:
            logger.error(
                f"Failed to load PyTorch BERT model {model_id}: {str(e)}")
            raise ValueError(f"Could not load BERT model {model_id}: {str(e)}")

    def get_tokenizer(self, model_id: str) -> Optional[PreTrainedTokenizer]:
        """
        Load or retrieve a tokenizer for BERT models.

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
        Load or retrieve a config for BERT models.

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
            logger.info("BERT registry cleared")
