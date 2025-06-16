from typing import Dict, Optional
from threading import Lock
import xgboost as xgb
import logging
import pickle
import os
from .base import NonTransformersModelRegistry
from jet.models.model_registry.base import ModelFeatures

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class XGBoostModelRegistry(NonTransformersModelRegistry):
    """Thread-safe registry for XGBoost models."""

    _instance = None
    _lock = Lock()
    _models: Dict[str, xgb.Booster] = {}

    def load_model(self, model_id: str, features: Optional[ModelFeatures] = None) -> Optional[xgb.Booster]:
        """
        Load or retrieve an XGBoost model by model_id (file path).

        Args:
            model_id: The file path to the saved XGBoost model.
            features: Optional configuration (ignored for XGBoost).

        Returns:
            Optional[xgb.Booster]: The loaded model instance.

        Raises:
            ValueError: If the model_id is invalid or loading fails.
        """
        with self._lock:
            if model_id in self._models:
                logger.info(
                    f"Reusing existing XGBoost model for model_id: {model_id}")
                return self._models[model_id]

            logger.info(f"Loading XGBoost model for model_id: {model_id}")
            try:
                with open(model_id, 'rb') as f:
                    model = pickle.load(f)
                if not isinstance(model, xgb.Booster):
                    raise ValueError(
                        f"Loaded model is not an XGBoost Booster: {type(model)}")
                self._models[model_id] = model
                return model
            except Exception as e:
                logger.error(
                    f"Failed to load XGBoost model {model_id}: {str(e)}")
                raise ValueError(
                    f"Could not load XGBoost model {model_id}: {str(e)}")

    def clear(self) -> None:
        """Clear all cached models."""
        with self._lock:
            self._models.clear()
            logger.info("XGBoost registry cleared")
