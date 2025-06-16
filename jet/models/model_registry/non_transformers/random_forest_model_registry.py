from typing import Dict, Optional
from threading import Lock
from sklearn.ensemble import RandomForestClassifier
import logging
import pickle
import os
from .base import NonTransformersModelRegistry
from jet.models.model_registry.base import ModelFeatures

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RandomForestModelRegistry(NonTransformersModelRegistry):
    """Thread-safe registry for Random Forest models."""

    _instance = None
    _lock = Lock()
    _models: Dict[str, RandomForestClassifier] = {}

    def load_model(self, model_id: str, features: Optional[ModelFeatures] = None) -> Optional[RandomForestClassifier]:
        """
        Load or retrieve a Random Forest model by model_id (file path).

        Args:
            model_id: The file path to the saved Random Forest model.
            features: Optional configuration (ignored for Random Forest).

        Returns:
            Optional[RandomForestClassifier]: The loaded model instance.

        Raises:
            ValueError: If the model_id is invalid or loading fails.
        """
        with self._lock:
            if model_id in self._models:
                logger.info(
                    f"Reusing existing Random Forest model for model_id: {model_id}")
                return self._models[model_id]

            logger.info(
                f"Loading Random Forest model for model_id: {model_id}")
            try:
                with open(model_id, 'rb') as f:
                    model = pickle.load(f)
                if not isinstance(model, RandomForestClassifier):
                    raise ValueError(
                        f"Loaded model is not a RandomForestClassifier: {type(model)}")
                self._models[model_id] = model
                return model
            except Exception as e:
                logger.error(
                    f"Failed to load Random Forest model {model_id}: {str(e)}")
                raise ValueError(
                    f"Could not load Random Forest model {model_id}: {str(e)}")

    def clear(self) -> None:
        """Clear all cached models."""
        with self._lock:
            self._models.clear()
            logger.info("Random Forest registry cleared")
