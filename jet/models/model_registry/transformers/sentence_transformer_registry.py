from typing import Dict, Optional, Tuple, Union, Literal, TypedDict
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ONNXSentenceTransformerWrapper:
    """Wrapper for ONNX SentenceTransformer model to mimic encode method."""

    def __init__(self, session: ort.InferenceSession, tokenizer: PreTrainedTokenizer):
        self.session = session
        self.tokenizer = tokenizer

    def encode(self, sentences, batch_size=32, **kwargs) -> np.ndarray:
        """
        Encode sentences to embeddings using the ONNX model.

        Args:
            sentences: List of strings to encode.
            batch_size: Batch size for encoding.
            **kwargs: Additional tokenizer arguments.

        Returns:
            np.ndarray: Embeddings for the input sentences.
        """
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
            embeddings = outputs[0]  # Assume first output is the embedding
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

    def load_model(self, model_id: str, features: Optional[ModelFeatures] = None) -> Optional[Union[SentenceTransformer, ONNXSentenceTransformerWrapper]]:
        """
        Load or retrieve a SentenceTransformer model by model_id.

        Args:
            model_id: The identifier of the model (e.g., 'all-MiniLM-L6-v2').
            features: Optional configuration (e.g., device, precision).

        Returns:
            Optional[Union[SentenceTransformer, ONNXSentenceTransformerWrapper]]: The loaded model instance.

        Raises:
            ValueError: If the model_id is invalid or loading fails.
        """
        features = features or {}
        with self._lock:
            if model_id in self._models:
                logger.info(
                    f"Reusing existing SentenceTransformer model for model_id: {model_id}")
                return self._models[model_id]

            if has_onnx_model_in_repo(model_id):
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
                f"No ONNX model found, loading PyTorch SentenceTransformer model for model_id: {model_id}")
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
        """
        Load or retrieve a tokenizer for SentenceTransformer models.

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
        Load or retrieve a config for SentenceTransformer models.

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
            logger.info("SentenceTransformer registry cleared")
