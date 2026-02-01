import numpy as np
from jet.adapters.llama_cpp.models import LLAMACPP_MODELS, LLAMACPP_MODELS_REVERSED
from jet.adapters.llama_cpp.types import (
    LLAMACPP_TYPES,
    EmbeddingVector,
)


def resolve_model_key(model: LLAMACPP_TYPES) -> str:
    """
    Retrieves the model key (short name) for a given model key or path.

    Args:
        model: A model key (short name) or full model path.

    Returns:
        The corresponding model key (short name).

    Raises:
        ValueError: If the model key or path is not recognized.
    """
    if model in LLAMACPP_MODELS:
        return model
    elif model in LLAMACPP_MODELS.values():
        return LLAMACPP_MODELS_REVERSED[model]
    for key, value in LLAMACPP_MODELS.items():
        if value == model:
            return key
    return model


def resolve_model_value(model: LLAMACPP_TYPES) -> str:
    """
    Retrieves the model value (full path) for a given model key or path.

    Args:
        model: A model key (short name) or full model path.

    Returns:
        The corresponding model value (full path).

    Raises:
        ValueError: If the model key or path is not recognized.
    """
    if model in LLAMACPP_MODELS:
        return LLAMACPP_MODELS[model]
    return model


def cosine_similarity(vec1: EmbeddingVector, vec2: EmbeddingVector) -> float:
    vec1 = np.asarray(vec1)
    vec2 = np.asarray(vec2)
    denom = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    if denom == 0.0:
        return 0.0
    return float(np.dot(vec1, vec2) / denom)
