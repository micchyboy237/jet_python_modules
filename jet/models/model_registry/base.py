from abc import ABC, abstractmethod
from typing import Optional, TypedDict, Literal
from threading import Lock
import logging
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelFeatures(TypedDict):
    """Configuration options for loading models."""
    device: Optional[Literal["cpu", "cuda", "mps"]]
    precision: Optional[Literal["fp16", "fp32"]]


class BaseModelRegistry(ABC):
    """Abstract base class for thread-safe model registries."""

    _instance = None
    _lock = Lock()

    def __new__(cls):
        """Ensure singleton behavior."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(BaseModelRegistry, cls).__new__(cls)
        return cls._instance

    @abstractmethod
    def load_model(self, model_id: str, features: Optional[ModelFeatures] = None) -> Optional[object]:
        """
        Load or retrieve a model by model_id.

        Args:
            model_id: The identifier of the model.
            features: Optional configuration (e.g., device, precision).

        Returns:
            Optional[object]: The loaded model instance.

        Raises:
            ValueError: If the model_id is invalid or loading fails.
        """
        pass

    @abstractmethod
    def get_tokenizer(self, model_id: str) -> Optional[object]:
        """
        Load or retrieve a tokenizer by model_id.

        Args:
            model_id: The identifier of the model.

        Returns:
            Optional[object]: The loaded tokenizer instance.

        Raises:
            ValueError: If the model_id is invalid or loading fails.
        """
        pass

    @abstractmethod
    def get_config(self, model_id: str) -> Optional[object]:
        """
        Load or retrieve a config by model_id.

        Args:
            model_id: The identifier of the model.

        Returns:
            Optional[object]: The loaded config instance.

        Raises:
            ValueError: If the model_id is invalid or loading fails.
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all cached models, tokenizers, and configs (for testing)."""
        pass

    def _select_device(self, features: ModelFeatures) -> str:
        if features.get("device") == "cuda" and torch.cuda.is_available():
            return "cuda"
        elif features.get("device") == "mps" and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
