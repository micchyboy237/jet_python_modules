from typing import Dict, Optional
from threading import Lock
from torchvision.models import resnet50, ResNet50_Weights
import torch
import logging
from .base import NonTransformersModelRegistry
from jet.models.model_registry.base import ModelFeatures

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResNetModelRegistry(NonTransformersModelRegistry):
    """Thread-safe registry for ResNet models."""

    _instance = None
    _lock = Lock()
    _models: Dict[str, torch.nn.Module] = {}

    def load_model(self, model_id: str, features: Optional[ModelFeatures] = None) -> Optional[torch.nn.Module]:
        """
        Load or retrieve a ResNet model by model_id.

        Args:
            model_id: The identifier of the model (e.g., 'resnet50').
            features: Optional configuration (e.g., device, precision).

        Returns:
            Optional[torch.nn.Module]: The loaded model instance.

        Raises:
            ValueError: If the model_id is invalid or loading fails.
        """
        features = features or {}
        with self._lock:
            if model_id in self._models:
                logger.info(
                    f"Reusing existing ResNet model for model_id: {model_id}")
                return self._models[model_id]

            logger.info(f"Loading ResNet model for model_id: {model_id}")
            try:
                if model_id != "resnet50":
                    raise ValueError(
                        f"Unsupported ResNet model_id: {model_id}")
                model = resnet50(weights=ResNet50_Weights.DEFAULT)
                device = self._select_device(features)
                model = model.to(device)
                if features.get("precision") == "fp16" and device != "cpu":
                    model = model.half()
                self._models[model_id] = model
                return model
            except Exception as e:
                logger.error(
                    f"Failed to load ResNet model {model_id}: {str(e)}")
                raise ValueError(
                    f"Could not load ResNet model {model_id}: {str(e)}")

    def clear(self) -> None:
        """Clear all cached models."""
        with self._lock:
            self._models.clear()
            logger.info("ResNet registry cleared")
