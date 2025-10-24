from jet.adapters.llama_cpp.models import LLAMACPP_MODELS, LLAMACPP_MODELS_REVERSED


def resolve_model_key(model: str) -> str:
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


def resolve_model_value(model: str) -> str:
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
