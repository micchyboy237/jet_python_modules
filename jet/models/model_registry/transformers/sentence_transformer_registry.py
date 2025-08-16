from pathlib import Path
from typing import Dict, List, Optional, OrderedDict, Tuple, Union, Literal, TypedDict
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoConfig, PreTrainedTokenizer, PretrainedConfig
import onnxruntime as ort
import numpy as np
import logging
import os
import torch

from jet.data.utils import generate_key
from jet.logger import logger
from jet.models.tokenizer.base import TokenizerWrapper, get_tokenizer_fn
from .base import BaseModelRegistry, ModelFeatures
from jet.models.onnx_model_checker import has_onnx_model_in_repo, get_onnx_model_paths
from jet.models.utils import get_context_size, get_embedding_size, resolve_model_value
from jet.models.model_types import EmbedModelType


class SentenceTransformerRegistry(BaseModelRegistry):
    """Registry for SentenceTransformer models with LRU cache and memory management."""
    _instance = None
    _max_cache_size = 3  # Limit to 3 models/tokenizers/sessions in cache
    _models: OrderedDict[str, SentenceTransformer] = OrderedDict()
    _tokenizers: OrderedDict[str, TokenizerWrapper] = OrderedDict()
    _configs: OrderedDict[str, PretrainedConfig] = OrderedDict()
    _onnx_sessions: OrderedDict[Tuple[str, str],
                                ort.InferenceSession] = OrderedDict()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(
                SentenceTransformerRegistry, cls).__new__(cls)
            # Initialize instance attributes
            cls._instance.model_id = None
            cls._instance.context_window = None
            cls._instance.dimensions = None
            cls._instance.truncate_dim = None
            cls._instance.max_length = None
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'model_id'):
            self.model_id = None
        if not hasattr(self, 'context_window'):
            self.context_window = None
        if not hasattr(self, 'dimensions'):
            self.dimensions = None
        if not hasattr(self, 'truncate_dim'):
            self.truncate_dim = None
        if not hasattr(self, 'max_length'):
            self.max_length = None

    @staticmethod
    def load_model(
        model_id: EmbedModelType = "static-retrieval-mrl-en-v1",
        truncate_dim: Optional[int] = None,
        prompts: Optional[dict[str, str]] = None,
        max_length: Optional[int] = None,
        device: Optional[Literal["cpu", "mps"]] = None
    ) -> SentenceTransformer:
        """Load or retrieve a SentenceTransformer model statically."""
        instance = SentenceTransformerRegistry()

        resolved_model_id = resolve_model_value(model_id)
        _cache_key = generate_key(resolved_model_id, truncate_dim)
        instance.model_id = resolved_model_id
        instance.context_window = get_context_size(resolved_model_id)
        instance.dimensions = get_embedding_size(resolved_model_id)
        instance.truncate_dim = truncate_dim
        instance.max_length = max_length or truncate_dim or instance.context_window
        if instance.max_length > instance.context_window:
            raise ValueError(
                f"max_length ({instance.max_length}) cannot be greater than context window ({instance.context_window})")

        # Check cache size and evict oldest if necessary
        if _cache_key not in instance._models and len(instance._models) >= instance._max_cache_size:
            oldest_key = next(iter(instance._models))
            del instance._models[oldest_key]
            logger.info(f"Evicted oldest model from cache: {oldest_key}")
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            gc.collect()

        if _cache_key in instance._models:
            logger.info(
                f"Reusing cached SentenceTransformer model for model_id: {resolved_model_id} | truncate_dim: {truncate_dim}")
            instance._models.move_to_end(_cache_key)  # Update LRU order
            return instance._models[_cache_key]

        logger.info(
            f"Loading SentenceTransformer model for model_id: {resolved_model_id} | truncate_dim: {truncate_dim}")
        try:
            model = instance._load_model(
                resolved_model_id, truncate_dim=truncate_dim, prompts=prompts, device=device)
            instance._models[_cache_key] = model
            instance._models.move_to_end(_cache_key)  # Update LRU order
            # Log memory usage
            process = psutil.Process()
            mem_usage = process.memory_info().rss / 1024 / 1024  # MB
            logger.debug(
                f"Memory usage after loading model: {mem_usage:.2f} MB")
            return model
        except Exception as e:
            logger.error(
                f"Failed to load SentenceTransformer model {resolved_model_id}: {str(e)}")
            raise ValueError(
                f"Could not load SentenceTransformer model {resolved_model_id}: {str(e)}")

    def _load_model(self, model_id: EmbedModelType, truncate_dim: Optional[int] = None, prompts: Optional[dict[str, str]] = None, **kwargs) -> Optional[SentenceTransformer]:
        device = kwargs.get("device", "cpu")
        try:
            logger.info(f"Loading embedding model on CPU (onnx)")
            model_instance = SentenceTransformer(
                model_id, device="cpu", backend="onnx", trust_remote_code=True, truncate_dim=truncate_dim, prompts=prompts,
            )
        except Exception as e:
            device = "mps" if torch.backends.mps.is_available(
            ) else "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(
                f"Loading embedding model on {device.upper()}: {model_id}")
            model_instance = SentenceTransformer(
                model_id, device=device, truncate_dim=truncate_dim, prompts=prompts,
            )
        # Ensure MPS memory is released
        if device == "mps":
            torch.mps.empty_cache()
        return model_instance

    @staticmethod
    def get_tokenizer(model_id: Optional[EmbedModelType] = None) -> TokenizerWrapper:
        instance = SentenceTransformerRegistry()
        resolved_model_id = resolve_model_value(model_id or instance.model_id)

        # Check tokenizer cache size
        if resolved_model_id not in instance._tokenizers and len(instance._tokenizers) >= instance._max_cache_size:
            oldest_key = next(iter(instance._tokenizers))
            del instance._tokenizers[oldest_key]
            logger.info(f"Evicted oldest tokenizer from cache: {oldest_key}")
            gc.collect()

        if resolved_model_id in instance._tokenizers:
            logger.info(f"Reusing tokenizer for model_id: {resolved_model_id}")
            instance._tokenizers.move_to_end(
                resolved_model_id)  # Update LRU order
            return instance._tokenizers[resolved_model_id]

        logger.info(f"Loading tokenizer for model_id: {resolved_model_id}")
        model = instance.load_model(resolved_model_id)
        tokenizer = model.tokenizer
        wrapped_tokenizer = TokenizerWrapper(
            tokenizer,
            max_length=instance.max_length,
            truncation=True
        )
        instance._tokenizers[resolved_model_id] = wrapped_tokenizer
        instance._tokenizers.move_to_end(resolved_model_id)  # Update LRU order
        return wrapped_tokenizer

    # def _load_tokenizer(self, **kwargs) -> TokenizerWrapper:
    #     kwargs = {
    #         "model_name_or_tokenizer": self.model_id,
    #         "disable_cache": True,
    #         "remove_pad_tokens": True,
    #         "max_length": self.max_length,
    #         **kwargs
    #     }
    #     tokenizer = get_tokenizer_fn(**kwargs)
    #     return tokenizer

    @staticmethod
    def get_config() -> Optional[PretrainedConfig]:
        instance = SentenceTransformerRegistry()

        resolved_model_id = resolve_model_value(instance.model_id)
        if resolved_model_id in instance._configs:
            logger.info(f"Reusing config for model_id: {resolved_model_id}")
            return instance._configs[resolved_model_id]
        logger.info(f"Loading config for model_id: {resolved_model_id}")
        try:
            model = SentenceTransformer(resolved_model_id)
            first_module = model._first_module()
            if hasattr(first_module, 'auto_model') and hasattr(first_module.auto_model, 'config'):
                config = first_module.auto_model.config
                instance._configs[resolved_model_id] = config
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
                instance._configs[resolved_model_id] = config
                return config
        except Exception as e:
            logger.error(
                f"Failed to load SentenceTransformer config {resolved_model_id}: {str(e)}")
            try:
                config = AutoConfig.from_pretrained(resolved_model_id)
                instance._configs[resolved_model_id] = config
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
                instance._configs[resolved_model_id] = config
                return config

    @staticmethod
    def clear() -> None:
        """Clear all cached models, tokenizers, and configs."""
        instance = SentenceTransformerRegistry()
        instance._models.clear()
        instance._tokenizers.clear()
        instance._configs.clear()
        instance._onnx_sessions.clear()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()
        logger.info("SentenceTransformer registry cleared")

    @staticmethod
    def generate_embeddings(
        input_data: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False,
        return_format: Literal["list", "numpy"] = "list",
    ) -> Union[List[float], List[List[float]], np.ndarray, torch.Tensor]:
        """Generate embeddings for input text using a SentenceTransformer model."""
        from jet.models.embeddings.base import generate_embeddings

        instance = SentenceTransformerRegistry()

        if instance.model_id is None:
            raise ValueError(
                "No model_id set. Please load a model using load_model first.")
        model = instance.load_model(
            model_id=instance.model_id,
            truncate_dim=instance.truncate_dim,
        )
        return generate_embeddings(
            input_data,
            model,
            batch_size,
            show_progress,
            return_format,
            truncate_dim=instance.truncate_dim,  # Explicitly pass truncate_dim
        )
