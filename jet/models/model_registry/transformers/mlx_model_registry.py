from abc import ABC
from typing import Optional, Dict, Literal
from threading import Lock
import logging
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import os
from transformers import AutoConfig, AutoTokenizer, PretrainedConfig, PreTrainedTokenizer
from .base import TransformersModelRegistry, ModelFeatures

logger = logging.getLogger(__name__)


class MLXModelRegistry(TransformersModelRegistry):
    """Abstract base class for MLX-based model registries."""
    _instance = None
    _lock = Lock()
    _models: Dict[str, nn.Module] = {}
    _tokenizers: Dict[str, PreTrainedTokenizer] = {}
    _configs: Dict[str, PretrainedConfig] = {}

    def load_model(self, model_id: str, features: Optional[ModelFeatures] = None) -> Optional[nn.Module]:
        """Load or retrieve an MLX model."""
        with self._lock:
            if model_id in self._models:
                logger.info(
                    f"Reusing existing MLX model for model_id: {model_id}")
                return self._models[model_id]
            logger.info(f"Loading MLX model for model_id: {model_id}")
            try:
                model = self._load_mlx_model(model_id, features or {})
                self._models[model_id] = model
                return model
            except Exception as e:
                logger.error(f"Failed to load MLX model {model_id}: {str(e)}")
                raise ValueError(
                    f"Could not load MLX model {model_id}: {str(e)}")

    def _load_mlx_model(self, model_id: str, features: ModelFeatures) -> nn.Module:
        """Load an MLX model with specified device and precision."""
        try:
            # Select device based on features
            # Default to MPS on Apple Silicon
            device = features.get("device", "mps")
            if device not in ["cpu", "mps"]:
                logger.error(
                    f"Unsupported device {device} for MLX model {model_id}")
                raise ValueError(
                    f"Device {device} is not supported for MLX (use 'cpu' or 'mps')")

            # Set precision
            precision = features.get("precision", "fp32")
            if precision not in ["fp16", "fp32"]:
                logger.error(
                    f"Unsupported precision {precision} for MLX model {model_id}")
                raise ValueError(
                    f"Precision {precision} is not supported for MLX (use 'fp16' or 'fp32')")

            # Configure MLX device
            if device == "cpu":
                mx.set_default_device(mx.cpu)
            else:  # mps
                mx.set_default_device(mx.gpu)

            # Configure precision
            dtype = mx.float16 if precision == "fp16" else mx.float32

            # Assume model weights are stored in a directory with model_id
            cache_dir = os.path.expanduser("~/.cache/mlx/models")
            model_path = os.path.join(cache_dir, model_id, "weights.npz")
            if not os.path.exists(model_path):
                logger.error(f"Model weights not found at {model_path}")
                raise FileNotFoundError(
                    f"Model weights for {model_id} not found at {model_path}")

            # Load weights
            weights = np.load(model_path, allow_pickle=True)
            model_params = {k: mx.array(v, dtype=dtype)
                            for k, v in weights.items()}

            # Create a placeholder model (subclasses should define actual architecture)
            class MLXModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.params = model_params

                def __call__(self, x):
                    # Placeholder forward pass
                    return x

            model = MLXModel()
            model.eval()  # Set to evaluation mode
            logger.info(
                f"Successfully loaded MLX model {model_id} on device {device} with {precision} precision")
            return model

        except Exception as e:
            logger.error(f"Failed to load MLX model {model_id}: {str(e)}")
            raise ValueError(f"Could not load MLX model {model_id}: {str(e)}")

    def get_tokenizer(self, model_id: str) -> Optional[PreTrainedTokenizer]:
        """Load or retrieve a tokenizer for the MLX model."""
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
        """Load or retrieve a config for the MLX model."""
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
            logger.info("MLX registry cleared")
