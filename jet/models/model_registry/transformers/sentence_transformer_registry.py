from typing import Dict, List, Optional, Tuple, Union, Literal, TypedDict
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoConfig, PreTrainedTokenizer, PretrainedConfig
import onnxruntime as ort
import numpy as np
import logging
import os
import torch

from jet.logger import logger
from .base import BaseModelRegistry, ModelFeatures
from jet.models.onnx_model_checker import has_onnx_model_in_repo, get_onnx_model_paths
from jet.models.utils import resolve_model_value


class SentenceTransformerRegistry(BaseModelRegistry):
    """Registry for SentenceTransformer models."""
    _instance = None
    _models: Dict[str, SentenceTransformer] = {}
    _tokenizers: Dict[str, PreTrainedTokenizer] = {}
    _configs: Dict[str, PretrainedConfig] = {}
    _onnx_sessions: Dict[Tuple[str, str], ort.InferenceSession] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(
                SentenceTransformerRegistry, cls).__new__(cls)
        return cls._instance

    @staticmethod
    def load_model(
        model_id: str = "static-retrieval-mrl-en-v1",
        device: Optional[Literal["cpu", "cuda", "mps"]] = "cpu",
        precision: Optional[Literal["fp32", "fp16"]] = "fp32",
        backend: Optional[Literal["pytorch", "onnx"]] = "onnx"
    ) -> SentenceTransformer:
        """Load or retrieve a SentenceTransformer model statically."""
        instance = SentenceTransformerRegistry()
        resolved_model_id = resolve_model_value(model_id)
        features = {"device": device,
                    "precision": precision, "backend": backend}
        if resolved_model_id in instance._models:
            logger.info(
                f"Reusing existing SentenceTransformer model for model_id: {resolved_model_id}")
            return instance._models[resolved_model_id]
        logger.info(
            f"Loading SentenceTransformer model for model_id: {resolved_model_id}")
        try:
            model = instance._load_model(resolved_model_id, features)
            instance._models[resolved_model_id] = model
            return model
        except Exception as e:
            logger.error(
                f"Failed to load SentenceTransformer model {resolved_model_id}: {str(e)}")
            raise ValueError(
                f"Could not load SentenceTransformer model {resolved_model_id}: {str(e)}")

    def _load_model(self, model_id: str, features: Optional[ModelFeatures] = None) -> Optional[SentenceTransformer]:
        features = features or {}
        if model_id in self._models:
            logger.info(
                f"Reusing existing SentenceTransformer model for model_id: {model_id}")
            return self._models[model_id]
        if features.get("backend", "pytorch") == "onnx" and has_onnx_model_in_repo(model_id):
            model = self._load_onnx_model(model_id, features)
            if model:
                return model
        logger.info(
            f"No ONNX model found or backend set to pytorch, loading PyTorch SentenceTransformer model for model_id: {model_id}")
        return self._load_pytorch_model(model_id, features)

    def _load_onnx_model(self, model_id: str, features: ModelFeatures) -> Optional[SentenceTransformer]:
        """Load an ONNX SentenceTransformer model."""
        if features.get("backend", "pytorch") != "onnx" or not has_onnx_model_in_repo(model_id):
            logger.debug(
                f"Debug: Skipping ONNX load for model_id={model_id}, backend={features.get('backend')}, has_onnx={has_onnx_model_in_repo(model_id)}")
            return None
        logger.info(
            f"Loading ONNX SentenceTransformer model for model_id: {model_id}")
        logger.debug(f"Debug: Checking ONNX model paths for {model_id}")
        try:
            onnx_paths = get_onnx_model_paths(model_id)
            logger.debug(f"Debug: ONNX paths found: {onnx_paths}")
            model = SentenceTransformer(model_id, device="cpu", backend="onnx")
            logger.debug(
                f"Debug: Model loaded with device={getattr(model, 'device', 'cpu')}")
            # Verify model with test encoding
            test_input = ["Test sentence"]
            logger.debug(
                f"Debug: Attempting test encoding with input={test_input}")
            test_embedding = model.encode(
                test_input, batch_size=1, convert_to_numpy=True)
            logger.debug(
                f"Debug: Test encoding successful, shape={test_embedding.shape}")
            self._models[model_id] = model
            return model
        except Exception as e:
            logger.error(
                f"Failed to load ONNX SentenceTransformer model {model_id}: {str(e)}")
            logger.debug(f"Debug: Exception details: {str(e)}")
            return None

    def _load_pytorch_model(self, model_id: str, features: ModelFeatures) -> SentenceTransformer:
        """Load a PyTorch SentenceTransformer model."""
        try:
            model = SentenceTransformer(model_id)
            device = self._select_device(features)
            model = model.to(device)
            if features.get("precision") == "fp16" and device != "cpu":
                model = model.half()
            self._models[model_id] = model
            return model
        except Exception as e:
            logger.error(
                f"Failed to load PyTorch SentenceTransformer model {model_id}: {str(e)}")
            raise ValueError(
                f"Could not load SentenceTransformer model {model_id}: {str(e)}")

    def get_tokenizer(self, model_id: str) -> Optional[PreTrainedTokenizer]:
        resolved_model_id = resolve_model_value(model_id)
        if resolved_model_id in self._tokenizers:
            logger.info(f"Reusing tokenizer for model_id: {resolved_model_id}")
            return self._tokenizers[resolved_model_id]
        logger.info(f"Loading tokenizer for model_id: {resolved_model_id}")
        try:
            if has_onnx_model_in_repo(resolved_model_id):
                logger.info(
                    f"Using SentenceTransformer with ONNX backend for tokenizer: {resolved_model_id}")
                model = SentenceTransformer(
                    resolved_model_id, device="cpu", backend="onnx")
                tokenizer = model.tokenizer
                self._tokenizers[resolved_model_id] = tokenizer
                return tokenizer
        except Exception as e:
            logger.warning(
                f"Failed to load SentenceTransformer with ONNX backend for tokenizer {resolved_model_id}: {str(e)}")
        try:
            model = SentenceTransformer(resolved_model_id)
            tokenizer = model.tokenizer
            self._tokenizers[resolved_model_id] = tokenizer
            return tokenizer
        except Exception as e:
            logger.error(
                f"Failed to load SentenceTransformer tokenizer {resolved_model_id}: {str(e)}")
            try:
                tokenizer = AutoTokenizer.from_pretrained(resolved_model_id)
                self._tokenizers[resolved_model_id] = tokenizer
                return tokenizer
            except Exception as e2:
                logger.error(
                    f"Failed to load AutoTokenizer {resolved_model_id}: {str(e2)}")
                raise ValueError(
                    f"Could not load tokenizer {resolved_model_id}: {str(e2)}")

    def get_config(self, model_id: str) -> Optional[PretrainedConfig]:
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
        model_id: str = "static-retrieval-mrl-en-v1",
        batch_size: int = 32,
        show_progress: bool = False,
        return_format: Literal["list", "numpy", "torch"] = "list",
        backend: Literal["pytorch", "onnx"] = "onnx"
    ) -> Union[List[float], List[List[float]], np.ndarray, torch.Tensor]:
        """Generate embeddings for input text using a SentenceTransformer model."""
        if return_format not in ["list", "numpy", "torch"]:
            raise ValueError(
                "return_format must be 'list', 'numpy', or 'torch'")
        resolved_model_id = resolve_model_value(model_id)
        logger.info(
            f"Generating embeddings for input type: {type(input_data)}, model: {resolved_model_id}, "
            f"show_progress: {show_progress}, return_format: {return_format}, backend: {backend}"
        )
        try:
            model = self._load_model(
                resolved_model_id, {"device": "cpu", "backend": backend})
            if not model:
                raise ValueError(f"Failed to load model {resolved_model_id}")
            logger.debug(
                f"Embedding model initialized with device: {getattr(model, 'device', 'cpu')}")
            if isinstance(input_data, str):
                logger.debug(
                    f"Processing single string input: {input_data[:50]}")
                embedding = model.encode(
                    input_data,
                    batch_size=1,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
                embedding = np.ascontiguousarray(embedding.astype(np.float32))
                logger.debug(f"Generated embedding shape: {embedding.shape}")
                if return_format == "numpy":
                    return embedding
                elif return_format == "torch":
                    return torch.tensor(embedding, dtype=torch.float32)
                return embedding.tolist()
            elif isinstance(input_data, list) and all(isinstance(item, str) for item in input_data):
                logger.debug(
                    f"Processing {len(input_data)} strings in batches of {batch_size}")
                logger.debug(
                    f"Debug: Input data sample: {input_data[:2] if input_data else []}")
                if not input_data:
                    logger.info(
                        "Empty input list, returning empty list of embeddings")
                    return [] if return_format == "list" else np.array([]) if return_format == "numpy" else torch.tensor([])
                embeddings = []
                iterator = range(0, len(input_data), batch_size)
                if show_progress:
                    from tqdm import tqdm
                    iterator = tqdm(iterator, desc="Embedding texts", total=len(
                        range(0, len(input_data), batch_size)))
                for i in iterator:
                    batch = input_data[i:i + batch_size]
                    logger.debug(
                        f"Debug: Encoding batch {i}-{min(i + batch_size, len(input_data))}, size: {len(batch)}")
                    try:
                        batch_embeddings = model.encode(
                            batch,
                            batch_size=len(batch),
                            convert_to_numpy=True,
                            normalize_embeddings=True
                        )
                        batch_embeddings = np.ascontiguousarray(
                            batch_embeddings.astype(np.float32))
                        embeddings.extend(batch_embeddings.tolist())
                        logger.debug(
                            f"Debug: Batch {i}-{min(i + batch_size, len(input_data))} encoded, shape: {batch_embeddings.shape}")
                        # Force garbage collection to prevent memory leaks
                        import gc
                        gc.collect()
                    except Exception as e:
                        logger.error(
                            f"Debug: Failed to encode batch {i}-{i + batch_size}: {str(e)}")
                        raise
                if return_format == "numpy":
                    return np.array(embeddings, dtype=np.float32)
                elif return_format == "torch":
                    return torch.tensor(embeddings, dtype=torch.float32)
                return embeddings
            else:
                logger.error(
                    f"Invalid input type: {type(input_data)}, expected str or List[str]")
                raise ValueError("Input must be a string or a list of strings")
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}")
            raise
