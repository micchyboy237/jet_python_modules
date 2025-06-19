from typing import Dict, List, Optional, Tuple, Union, Literal, TypedDict
from threading import Lock
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoConfig, PreTrainedTokenizer, PretrainedConfig
import onnxruntime as ort
import numpy as np
import logging
import os
import torch
from .base import BaseModelRegistry, ModelFeatures
from jet.models.onnx_model_checker import has_onnx_model_in_repo, get_onnx_model_paths
from jet.models.utils import resolve_model_value

logger = logging.getLogger(__name__)


class ONNXSentenceTransformerWrapper:
    """Wrapper for ONNX SentenceTransformer model to mimic encode method."""

    def __init__(self, session: ort.InferenceSession, tokenizer: PreTrainedTokenizer):
        self.session = session
        self.tokenizer = tokenizer

    def encode(self, sentences, batch_size=32, **kwargs) -> np.ndarray:
        logger.info(
            f"Encoding {len(sentences)} sentences with ONNX SentenceTransformer model")
        all_embeddings = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
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
            embeddings = outputs[0]
            all_embeddings.append(embeddings)
        return np.concatenate(all_embeddings, axis=0)


class SentenceTransformerRegistry(BaseModelRegistry):
    """Thread-safe registry for SentenceTransformer models."""
    _instance = None
    _lock = Lock()
    _models: Dict[str, Union[SentenceTransformer,
                             ONNXSentenceTransformerWrapper]] = {}
    _tokenizers: Dict[str, PreTrainedTokenizer] = {}
    _configs: Dict[str, PretrainedConfig] = {}
    _onnx_sessions: Dict[Tuple[str, str], ort.InferenceSession] = {}

    @staticmethod
    def load_model(
        model_id: str = "static-retrieval-mrl-en-v1",
        device: Optional[Literal["cpu", "cuda", "mps"]] = "cpu",
        precision: Optional[Literal["fp32", "fp16"]] = "fp32",
        backend: Optional[Literal["pytorch", "onnx"]] = "pytorch"
    ) -> Union[SentenceTransformer, ONNXSentenceTransformerWrapper]:
        """Load or retrieve a SentenceTransformer model statically."""
        instance = SentenceTransformerRegistry()
        resolved_model_id = resolve_model_value(model_id)
        features = {"device": device,
                    "precision": precision, "backend": backend}

        with instance._lock:
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

    def _load_model(self, model_id: str, features: Optional[ModelFeatures] = None) -> Optional[Union[SentenceTransformer, ONNXSentenceTransformerWrapper]]:
        features = features or {}
        with self._lock:
            if model_id in self._models:
                logger.info(
                    f"Reusing existing SentenceTransformer model for model_id: {model_id}")
                return self._models[model_id]

            if features.get("backend", "pytorch") == "onnx" and has_onnx_model_in_repo(model_id):
                onnx_paths = get_onnx_model_paths(model_id)
                if onnx_paths:
                    selected_path = None
                    for path in onnx_paths:
                        if "arm64.onnx" in path:
                            selected_path = path
                            break
                        elif "quantized" in path:
                            selected_path = path
                        elif "model.onnx" in path and not selected_path:
                            selected_path = path
                    if selected_path:
                        session_key = (model_id, selected_path)
                        if session_key in self._onnx_sessions:
                            logger.info(
                                f"Reusing ONNX session for model_id: {model_id}, path: {selected_path}")
                            session = self._onnx_sessions[session_key]
                        else:
                            logger.info(
                                f"Loading ONNX SentenceTransformer model for model_id: {model_id}, path: {selected_path}")
                            try:
                                full_path = os.path.join(
                                    "/Users/jethroestrada/.cache/huggingface/hub", selected_path)
                                session_options = ort.SessionOptions()
                                providers = ["CPUExecutionProvider"]
                                if features.get("device") == "cuda":
                                    providers = [
                                        "CUDAExecutionProvider"] + providers
                                session = ort.InferenceSession(
                                    full_path, providers=providers, sess_options=session_options)
                                self._onnx_sessions[session_key] = session
                            except Exception as e:
                                logger.error(
                                    f"Failed to load ONNX SentenceTransformer model {model_id}: {str(e)}")
                                return self._load_pytorch_model(model_id, features)
                        try:
                            tokenizer = self.get_tokenizer(model_id)
                            model = ONNXSentenceTransformerWrapper(
                                session, tokenizer)
                            self._models[model_id] = model
                            return model
                        except Exception as e:
                            logger.error(
                                f"Failed to load tokenizer for ONNX SentenceTransformer model {model_id}: {str(e)}")
                            return self._load_pytorch_model(model_id, features)

            logger.info(
                f"No ONNX model found or backend set to pytorch, loading PyTorch SentenceTransformer model for model_id: {model_id}")
            return self._load_pytorch_model(model_id, features)

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
            logger.info("SentenceTransformer registry cleared")

    def generate_embeddings(
        self,
        input_data: Union[str, List[str]],
        model_id: str = "static-retrieval-mrl-en-v1",
        batch_size: int = 32,
        show_progress: bool = False,
        return_format: Literal["list", "numpy", "torch"] = "list",
        backend: Literal["pytorch", "onnx"] = "pytorch"
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
            logger.debug(
                f"Embedding model initialized with device: {getattr(model, 'device', 'cpu')}")

            if isinstance(input_data, str):
                logger.debug(
                    f"Processing single string input: {input_data[:50]}")
                embedding = model.encode(
                    input_data, batch_size=batch_size, convert_to_numpy=True)
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
                        f"Encoding batch {i}-{min(i + batch_size, len(input_data))}")
                    batch_embeddings = model.encode(
                        batch, batch_size=batch_size, convert_to_numpy=True)
                    batch_embeddings = np.ascontiguousarray(
                        batch_embeddings.astype(np.float32))
                    embeddings.extend(batch_embeddings.tolist())

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
