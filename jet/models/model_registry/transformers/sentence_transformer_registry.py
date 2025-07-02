from typing import Dict, List, Optional, Tuple, Union, Literal, TypedDict
from llguidance import TokenizerWrapper
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoConfig, PreTrainedTokenizer, PretrainedConfig
import onnxruntime as ort
import numpy as np
import logging
import os
import torch

from jet.data.utils import generate_key
from jet.logger import logger
from jet.models.tokenizer.base import get_tokenizer_fn
from .base import BaseModelRegistry, ModelFeatures
from jet.models.onnx_model_checker import has_onnx_model_in_repo, get_onnx_model_paths
from jet.models.utils import resolve_model_value
from jet.models.model_types import EmbedModelType


class SentenceTransformerRegistry(BaseModelRegistry):
    """Registry for SentenceTransformer models."""
    _instance = None
    _models: Dict[str, SentenceTransformer] = {}
    _tokenizers: Dict[str, TokenizerWrapper] = {}
    _configs: Dict[str, PretrainedConfig] = {}
    _onnx_sessions: Dict[Tuple[str, str], ort.InferenceSession] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(
                SentenceTransformerRegistry, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        self.model_id: Optional[EmbedModelType] = None

    @staticmethod
    def load_model(
        model_id: EmbedModelType = "static-retrieval-mrl-en-v1",
        truncate_dim: Optional[int] = None,
    ) -> SentenceTransformer:
        """Load or retrieve a SentenceTransformer model statically."""
        instance = SentenceTransformerRegistry()

        resolved_model_id = resolve_model_value(model_id)
        _cache_key = generate_key(resolved_model_id, truncate_dim)
        instance.model_id = resolved_model_id  # Set instance model_id
        if _cache_key in instance._models:
            logger.info(
                f"Reusing existing SentenceTransformer model for model_id: {resolved_model_id}")
            return instance._models[_cache_key]
        logger.info(
            f"Loading SentenceTransformer model for model_id: {resolved_model_id}")
        try:
            model = instance._load_model(resolved_model_id)
            instance._models[_cache_key] = model
            return model
        except Exception as e:
            logger.error(
                f"Failed to load SentenceTransformer model {resolved_model_id}: {str(e)}")
            raise ValueError(
                f"Could not load SentenceTransformer model {resolved_model_id}: {str(e)}")

    def _load_model(self, model_id: EmbedModelType, truncate_dim: Optional[int] = None) -> Optional[SentenceTransformer]:
        try:
            logger.info(f"Loading embedding model on CPU (onnx): {model_id}")
            model_instance = SentenceTransformer(
                model_id, device="cpu", backend="onnx", truncate_dim=truncate_dim,
                model_kwargs={'file_name': 'model.onnx', 'subfolder': 'onnx'})
        except Exception as e:
            logger.warning(f"Falling back to MPS for embed model due to: {e}")
            model_instance = SentenceTransformer(model_id, device="mps")
        return model_instance

    def get_tokenizer(self, model_id: EmbedModelType) -> TokenizerWrapper:
        resolved_model_id = resolve_model_value(model_id)
        if resolved_model_id in SentenceTransformerRegistry._tokenizers:
            logger.info(
                f"Reusing tokenizer for model_id: {resolved_model_id}")
            return SentenceTransformerRegistry._tokenizers[resolved_model_id]

        logger.info(f"Loading tokenizer for model_id: {resolved_model_id}")
        tokenizer = get_tokenizer_fn(model_id, disable_cache=True)
        SentenceTransformerRegistry._tokenizers[resolved_model_id] = tokenizer
        return tokenizer

    def get_config(self, model_id: EmbedModelType) -> Optional[PretrainedConfig]:
        resolved_model_id = resolve_model_value(model_id)
        if resolved_model_id in self._configs:
            logger.info(f"Reusing config for model_id: {resolved_model_id}")
            return self._configs[resolved_model_id]
        logger.info(f"Loading config for model_id: {resolved_model_id}")
        try:
            model = SentenceTransformer(resolved_model_id)
            first_module = model._first_module()
            if hasattr(first_module, 'auto_model') and hasattr(first_module.auto_model, 'config'):
                config = first_module.auto_model.config
                self._configs[resolved_model_id] = config
                return config
            else:
                from transformers import PretrainedConfig
                config_dict = {
                    "model_type": "sentence_transformer",
                    "hidden_size": model.get_sentence_embedding_dimension(),
                    "max_seq_length": model.max_seq_length,
                    "tokenizer_class": model.tokenizer.__class__.__name__,
                }
                config = PretrainedConfig(**config_dict)
                self._configs[resolved_model_id] = config
                return config
        except Exception as e:
            logger.error(
                f"Failed to load SentenceTransformer config {resolved_model_id}: {str(e)}")
            try:
                config = AutoConfig.from_pretrained(resolved_model_id)
                self._configs[resolved_model_id] = config
                return config
            except Exception as e2:
                logger.error(
                    f"Failed to load AutoConfig {resolved_model_id}: {str(e2)}")
                from transformers import PretrainedConfig
                config_dict = {
                    "model_type": "sentence_transformer",
                    "hidden_size": 1024,
                    "max_seq_length": 512,
                }
                config = PretrainedConfig(**config_dict)
                self._configs[resolved_model_id] = config
                return config

    def clear(self) -> None:
        """Clear all cached models, tokenizers, and configs."""
        self._models.clear()
        self._tokenizers.clear()
        self._configs.clear()
        self._onnx_sessions.clear()
        logger.info("SentenceTransformer registry cleared")

    def generate_embeddings(
        self,
        input_data: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False,
        return_format: Literal["list", "numpy"] = "list",
    ) -> Union[List[float], List[List[float]], np.ndarray, torch.Tensor]:
        """Generate embeddings for input text using a SentenceTransformer model."""
        from jet.models.embeddings.base import generate_embeddings

        if self.model_id is None:
            raise ValueError(
                "No model_id set. Please load a model using load_model first.")
        model = self.load_model(
            model_id=self.model_id,
        )
        return generate_embeddings(input_data, model, batch_size, show_progress, return_format)
