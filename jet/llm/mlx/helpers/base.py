from typing import Tuple
from datetime import datetime
from jet.llm.mlx.mlx_types import MLXTokenizer, ModelType
from jet.llm.mlx.models import resolve_model
import mlx.nn as nn


def load_model(model: ModelType) -> Tuple[nn.Module, MLXTokenizer]:
    from mlx_lm import load

    model_path = str(resolve_model(model))

    return load(model_path)


def get_system_date_prompt():
    return f"Today's date is {datetime.now().strftime('%B %d, %Y')}"


__all__ = [
    "load_model",
]
