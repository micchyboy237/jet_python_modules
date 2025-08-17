from mlx_lm import load
from mlx_lm.utils import TokenizerWrapper
from jet.llm.mlx.models import resolve_model
from jet.models.model_types import LLMModelType


class ModelLoadError(Exception):
    """Raised when model or tokenizer loading fails."""
    pass


class ModelComponents:
    """Encapsulates model and tokenizer for easier management."""

    def __init__(self, model, tokenizer: TokenizerWrapper):
        self.model = model
        self.tokenizer = tokenizer


def load_model_components(model_path: LLMModelType) -> ModelComponents:
    """Loads model and tokenizer from the specified path."""
    try:
        model, tokenizer = load(resolve_model(model_path))
        return ModelComponents(model, tokenizer)
    except Exception as e:
        raise ModelLoadError(f"Error loading model or tokenizer: {e}")
