from abc import ABC
from typing import Optional, Dict, Tuple
from threading import Lock
import onnxruntime as ort
import os
import logging
from jet.models.config import MODELS_CACHE_DIR
from jet.models.model_registry.base import BaseModelRegistry, ModelFeatures
from jet.models.onnx_model_checker import has_onnx_model_in_repo, get_onnx_model_paths

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TransformersModelRegistry(BaseModelRegistry, ABC):
    """Abstract base class for transformer-based model registries."""

    _onnx_sessions: Dict[Tuple[str, str], ort.InferenceSession] = {}

    def _load_onnx_session(self, model_id: str, features: ModelFeatures) -> Optional[ort.InferenceSession]:
        """Load or retrieve an ONNX session for the model."""
        if not has_onnx_model_in_repo(model_id):
            return None

        onnx_paths = get_onnx_model_paths(model_id)
        if not onnx_paths:
            return None

        selected_path = None
        for path in onnx_paths:
            if "arm64.onnx" in path:
                selected_path = path
                break
            elif "quantized" in path:
                selected_path = path
            elif "model.onnx" in path and not selected_path:
                selected_path = path

        if not selected_path:
            return None

        session_key = (model_id, selected_path)
        with self._lock:
            if session_key in self._onnx_sessions:
                logger.info(
                    f"Reusing ONNX session for model_id: {model_id}, path: {selected_path}")
                return self._onnx_sessions[session_key]

            logger.info(
                f"Loading ONNX model for model_id: {model_id}, path: {selected_path}")
            try:
                cache_dir = MODELS_CACHE_DIR
                full_path = os.path.join(cache_dir, selected_path)
                session_options = ort.SessionOptions()
                providers = ["CPUExecutionProvider"]
                if features.get("device") == "cuda":
                    providers = ["CUDAExecutionProvider"] + providers
                session = ort.InferenceSession(
                    full_path, providers=providers, sess_options=session_options)
                self._onnx_sessions[session_key] = session
                return session
            except Exception as e:
                logger.error(f"Failed to load ONNX model {model_id}: {str(e)}")
                return None
