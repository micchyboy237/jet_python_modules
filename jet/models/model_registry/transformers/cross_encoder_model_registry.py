from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Literal, TypedDict
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer, AutoConfig, PreTrainedTokenizer, PretrainedConfig
import onnxruntime as ort
import numpy as np
import logging
import os
import torch

from jet.data.utils import generate_key
from jet.logger import logger
from jet.models.model_registry.base import BaseModelRegistry
from jet.models.tokenizer.base import TokenizerWrapper, get_tokenizer_fn
from jet.models.utils import get_context_size, resolve_model_value
from jet.models.model_types import RerankModelType


class CrossEncoderRegistry(BaseModelRegistry):
    """Registry for CrossEncoder models."""
    _instance = None
    _models: Dict[str, CrossEncoder] = {}
    _tokenizers: Dict[str, TokenizerWrapper] = {}
    _configs: Dict[str, PretrainedConfig] = {}
    _onnx_sessions: Dict[Tuple[str, str], ort.InferenceSession] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CrossEncoderRegistry, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'model_id'):
            self.model_id = None
        if not hasattr(self, 'context_window'):
            self.context_window = None
        if not hasattr(self, 'max_length'):
            self.max_length = None

    @staticmethod
    def load_model(
        model_id: RerankModelType = "cross-encoder/ms-marco-MiniLM-L6-v2",
        max_length: Optional[int] = None,
        device: Optional[Literal["cpu", "mps"]] = None
    ) -> CrossEncoder:
        """Load or retrieve a CrossEncoder model statically."""
        instance = CrossEncoderRegistry()

        resolved_model_id = resolve_model_value(model_id)
        # No truncate_dim for CrossEncoder
        _cache_key = generate_key(resolved_model_id, None)
        instance.model_id = resolved_model_id
        instance.context_window = get_context_size(resolved_model_id)
        instance.max_length = max_length or instance.context_window
        if instance.max_length > instance.context_window:
            raise ValueError(
                f"max_length ({instance.max_length}) cannot be greater than context window ({instance.context_window})")
        if _cache_key in instance._models:
            logger.info(
                f"Reusing existing CrossEncoder model for model_id: {resolved_model_id}")
            return instance._models[_cache_key]
        logger.info(
            f"Loading CrossEncoder model for model_id: {resolved_model_id}")
        try:
            model = instance._load_model(
                resolved_model_id, max_length=max_length, device=device)
            instance._models[_cache_key] = model
            return model
        except Exception as e:
            logger.error(
                f"Failed to load CrossEncoder model {resolved_model_id}: {str(e)}")
            raise ValueError(
                f"Could not load CrossEncoder model {resolved_model_id}: {str(e)}")

    def _load_model(self, model_id: RerankModelType, max_length: Optional[int] = None, **kwargs) -> Optional[CrossEncoder]:
        try:
            device = "mps" if torch.backends.mps.is_available(
            ) else "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(
                f"Loading embedding model on {device.upper()}: {model_id}")
            model_instance = CrossEncoder(
                model_id, device=device, max_length=max_length)
        except Exception as e:
            logger.warning(
                f"Falling back to CPU (onnx) for CrossEncoder model due to: {e}")
            model_instance = CrossEncoder(
                model_id, device="cpu", backend="onnx", trust_remote_code=True, max_length=max_length,
                # model_kwargs={'file_name': 'model.onnx', 'subfolder': 'onnx'}
            )
        return model_instance

    @staticmethod
    def get_tokenizer() -> TokenizerWrapper:
        instance = CrossEncoderRegistry()
        resolved_model_id = resolve_model_value(instance.model_id)
        if resolved_model_id in CrossEncoderRegistry._tokenizers:
            logger.info(f"Reusing tokenizer for model_id: {resolved_model_id}")
            return CrossEncoderRegistry._tokenizers[resolved_model_id]
        logger.info(f"Loading tokenizer for model_id: {resolved_model_id}")
        tokenizer = instance._load_tokenizer()
        CrossEncoderRegistry._tokenizers[resolved_model_id] = tokenizer
        return tokenizer

    def _load_tokenizer(self, **kwargs) -> TokenizerWrapper:
        kwargs = {
            "model_name_or_tokenizer": self.model_id,
            "disable_cache": True,
            "remove_pad_tokens": True,
            "max_length": self.max_length,
            **kwargs
        }
        tokenizer = get_tokenizer_fn(**kwargs)
        return tokenizer

    @staticmethod
    def get_config() -> Optional[PretrainedConfig]:
        instance = CrossEncoderRegistry()
        resolved_model_id = resolve_model_value(instance.model_id)
        if resolved_model_id in instance._configs:
            logger.info(f"Reusing config for model_id: {resolved_model_id}")
            return instance._configs[resolved_model_id]
        logger.info(f"Loading config for model_id: {resolved_model_id}")
        try:
            config = AutoConfig.from_pretrained(resolved_model_id)
            instance._configs[resolved_model_id] = config
            return config
        except Exception as e:
            logger.error(
                f"Failed to load AutoConfig {resolved_model_id}: {str(e)}")
            config_dict = {
                "model_type": "cross_encoder",
                "max_seq_length": instance.max_length or 512,
            }
            config = PretrainedConfig(**config_dict)
            instance._configs[resolved_model_id] = config
            return config

    @staticmethod
    def clear() -> None:
        """Clear all cached models, tokenizers, and configs."""
        instance = CrossEncoderRegistry()
        instance._models.clear()
        instance._tokenizers.clear()
        instance._configs.clear()
        instance._onnx_sessions.clear()
        logger.info("CrossEncoder registry cleared")

    @staticmethod
    def predict_scores(
        input_pairs: Union[Tuple[str, str], List[Tuple[str, str]]],
        batch_size: int = 32,
        show_progress: bool = False,
        return_format: Literal["list", "numpy"] = "list",
    ) -> Union[float, List[float], np.ndarray]:
        """Generate similarity scores for input sentence pairs using a CrossEncoder model."""
        instance = CrossEncoderRegistry()
        if instance.model_id is None:
            raise ValueError(
                "No model_id set. Please load a model using load_model first.")
        model = instance.load_model(
            model_id=instance.model_id, max_length=instance.max_length)

        # Normalize input to list of tuples
        input_pairs = [input_pairs] if isinstance(
            input_pairs, tuple) else input_pairs

        # Generate predictions
        predictions = model.predict(
            input_pairs,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=(return_format == "numpy")
        )

        # Handle return format
        if return_format == "numpy":
            return predictions
        return predictions.tolist() if isinstance(predictions, np.ndarray) else predictions
