import numpy as np
from jet.adapters.llama_cpp.models import (
    LLAMACPP_MODEL_CONTEXTS,
    LLAMACPP_MODEL_EMBEDDING_SIZES,
    LLAMACPP_MODELS,
    LLAMACPP_MODELS_REVERSED,
)
from jet.adapters.llama_cpp.types import (
    LLAMACPP_KEYS,
    LLAMACPP_TYPES,
    LLAMACPP_VALUES,
    EmbeddingVector,
)


def resolve_model_key(model: LLAMACPP_TYPES) -> LLAMACPP_KEYS:
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


def resolve_model_value(model: LLAMACPP_TYPES) -> LLAMACPP_VALUES:
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


def get_embedding_size(model: LLAMACPP_KEYS) -> int:
    """
    Returns the embedding size (hidden dimension) for the given model key or full model path.

    Args:
        model: A model key or model path.

    Returns:
        The embedding size (hidden dimension).

    Raises:
        ValueError: If the model is not recognized or missing an embedding size.
    """
    model_key = resolve_model_key(model)
    if model_key not in LLAMACPP_MODEL_EMBEDDING_SIZES:
        error_msg = f"Missing embedding size for model: {model_key}"
        raise ValueError(error_msg)
    return LLAMACPP_MODEL_EMBEDDING_SIZES[model_key]


def get_context_size(model: LLAMACPP_KEYS) -> int:
    """
    Returns the context size (hidden dimension) for the given model key or full model path.

    Args:
        model: A model key or model path.

    Returns:
        The maximum context size.

    Raises:
        ValueError: If the model is not recognized or missing an context size.
    """
    model_key = resolve_model_key(model)
    if model_key not in LLAMACPP_MODEL_CONTEXTS:
        error_msg = f"Missing context size for model: {model_key}"
        raise ValueError(error_msg)
    return LLAMACPP_MODEL_CONTEXTS[model_key]
