from abc import ABC
from typing import Optional, Dict, Literal
from threading import Lock
from jet.logger import logger
from jet.models.convert import convert_hf_to_mlx
from jet.models.utils import get_local_repo_dir
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from transformers import AutoConfig, PretrainedConfig
from pathlib import Path
from tokenizers import Tokenizer
from .base import TransformersModelRegistry, ModelFeatures


class MLXModelRegistry(TransformersModelRegistry):
    """Abstract base class for MLX-based model registries."""
    _instance = None
    _lock = Lock()
    _models: Dict[str, nn.Module] = {}
    _tokenizers: Dict[str, Tokenizer] = {}
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
            device = features.get("device", "mps")
            if device not in ["cpu", "mps"]:
                logger.error(
                    f"Unsupported device {device} for MLX model {model_id}")
                raise ValueError(
                    f"Device {device} is not supported for MLX (use 'cpu' or 'mps')")
            precision = features.get("precision", "fp32")
            if precision not in ["fp16", "fp32"]:
                logger.error(
                    f"Unsupported precision {precision} for MLX model {model_id}")
                raise ValueError(
                    f"Precision {precision} is not supported for MLX (use 'fp16' or 'fp32')")
            if device == "cpu":
                mx.set_default_device(mx.cpu)
            else:
                mx.set_default_device(mx.gpu)
            dtype = mx.float16 if precision == "fp16" else mx.float32
            cache_dir = Path(get_local_repo_dir(model_id))
            snapshot_dir = cache_dir / "snapshots"
            model_path = next((p for p in snapshot_dir.glob(
                "*/weights.npz") if p.is_file()), None)
            if not model_path or not model_path.exists():
                logger.info(
                    f"weights.npz not found in {snapshot_dir}. Attempting to convert from safetensors.")
                weights_dir = cache_dir / "weights"
                safetensors_path = next((p.parent for p in snapshot_dir.glob(
                    "*/model.safetensors") if p.is_file()), None)
                if not safetensors_path:
                    logger.error(
                        f"No safetensors files found in {snapshot_dir}")
                    raise FileNotFoundError(
                        f"No model.safetensors or model.safetensors.index.json found in {snapshot_dir}")
                convert_hf_to_mlx(
                    hf_path=str(safetensors_path),
                    weights_dir=str(weights_dir),
                    quantize=features.get("quantize", False),
                    q_group_size=features.get("q_group_size", 64),
                    q_bits=features.get("q_bits", 4),
                    dtype="float32" if precision == "fp32" else "float16",
                    quant_predicate=features.get("quant_predicate", None),
                    revision=None,
                    overwrite=True,
                    model_id=model_id,  # Pass model_id to convert_hf_to_mlx
                )
                model_path = weights_dir / "weights.npz"
                if not model_path.exists():
                    logger.error(
                        f"Failed to generate weights.npz at {model_path}")
                    raise FileNotFoundError(
                        f"Model weights for {model_id} not found and could not be converted in {snapshot_dir}. "
                        f"Please ensure the model weights are downloaded and compatible."
                    )
            weights = np.load(model_path, allow_pickle=True)
            model_params = {k: mx.array(v, dtype=dtype)
                            for k, v in weights.items()}

            class MLXModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.params = model_params

                def __call__(self, x):
                    return x
            model = MLXModel()
            model.eval()
            logger.info(
                f"Successfully loaded MLX model {model_id} on device {device} with {precision} precision")
            return model
        except Exception as e:
            logger.error(f"Failed to load MLX model {model_id}: {str(e)}")
            raise ValueError(f"Could not load MLX model {model_id}: {str(e)}")

    def get_tokenizer(self, model_id: str) -> Optional[Tokenizer]:
        """Load or retrieve a tokenizer for the MLX model."""
        with self._lock:
            if model_id in self._tokenizers:
                logger.info(f"Reusing tokenizer for model_id: {model_id}")
                return self._tokenizers[model_id]
            logger.info(f"Loading tokenizer for model_id: {model_id}")
            try:
                cache_dir = Path(get_local_repo_dir(model_id))
                snapshot_dir = cache_dir / "snapshots"
                tokenizer_files = list(snapshot_dir.glob("*/tokenizer.json"))
                logger.debug(
                    f"Found {len(tokenizer_files)} tokenizer.json files in {snapshot_dir}")
                for tokenizer_path in tokenizer_files:
                    try:
                        resolved_path = tokenizer_path.resolve()
                        logger.debug(
                            f"Resolved {tokenizer_path} to {resolved_path}")
                        if not resolved_path.is_file():
                            logger.warning(
                                f"Resolved path is not a file: {resolved_path}")
                            continue
                        tokenizer = Tokenizer.from_file(str(resolved_path))
                        logger.info(
                            f"Successfully loaded tokenizer from local cache: {resolved_path}")
                        self._tokenizers[model_id] = tokenizer
                        return tokenizer
                    except Exception as local_e:
                        logger.error(
                            f"Failed to load tokenizer from {resolved_path}: {str(local_e)}")
                        continue
                error_msg = f"Could not load tokenizer for {model_id} from local cache."
                logger.error(error_msg)
                raise ValueError(error_msg)
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
                cache_dir = Path(get_local_repo_dir(model_id))
                snapshot_dir = cache_dir / "snapshots"
                config_files = list(snapshot_dir.glob("*/config.json"))
                logger.debug(
                    f"Found {len(config_files)} config.json files in {snapshot_dir}")
                for config_path in config_files:
                    try:
                        resolved_path = config_path.resolve()
                        logger.debug(
                            f"Resolved {config_path} to {resolved_path}")
                        if not resolved_path.is_file():
                            logger.warning(
                                f"Resolved path is not a file: {resolved_path}")
                            continue
                        config = AutoConfig.from_pretrained(str(resolved_path))
                        logger.info(
                            f"Successfully loaded config from local cache: {resolved_path}")
                        self._configs[model_id] = config
                        return config
                    except Exception as local_e:
                        logger.error(
                            f"Failed to load config from {resolved_path}: {str(local_e)}")
                        continue
                error_msg = f"Could not load config for {model_id} from local cache."
                logger.error(error_msg)
                raise ValueError(error_msg)
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
