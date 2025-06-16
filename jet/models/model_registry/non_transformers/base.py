from abc import ABC
from typing import Optional, Dict
from threading import Lock
import logging
from jet.models.model_registry.base import BaseModelRegistry, ModelFeatures

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NonTransformersModelRegistry(BaseModelRegistry, ABC):
    """Abstract base class for non-transformer-based model registries."""

    _models: Dict[str, object] = {}

    def get_tokenizer(self, model_id: str) -> Optional[object]:
        """
        Non-transformer models typically don't use tokenizers.

        Args:
            model_id: The identifier of the model.

        Returns:
            None: Non-transformer models do not use tokenizers.

        Raises:
            ValueError: If tokenizer is requested for non-transformer model.
        """
        logger.warning(
            f"Tokenizers are not supported for non-transformer model_id: {model_id}")
        raise ValueError(
            f"Tokenizers are not applicable for non-transformer model {model_id}")

    def get_config(self, model_id: str) -> Optional[object]:
        """
        Non-transformer models typically don't use configs.

        Args:
            model_id: The identifier of the model.

        Returns:
            None: Non-transformer models do not use configs.

        Raises:
            ValueError: If config is requested for non-transformer model.
        """
        logger.warning(
            f"Configs are not supported for non-transformer model_id: {model_id}")
        raise ValueError(
            f"Configs are not applicable for non-transformer model {model_id}")
