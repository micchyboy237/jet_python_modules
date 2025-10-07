LLAMACPP_EMBED_MODELS = {
    "embeddinggemma": "google/embeddinggemma-300m",
    "nomic-embed-text": "nomic-ai/nomic-embed-text-v1.5",
    "nomic-embed-text-v2-moe": "nomic-ai/nomic-embed-text-v2-moe",
}

LLAMACPP_EMBED_MODELS_GGUF_MAPPING = {
    "embeddinggemma": "embeddinggemma-300M-Q8_0.gguf",
    "nomic-embed-text": "nomic-embed-text-v1.5.Q4_K_M.gguf",
    "nomic-embed-text-v2-moe": "nomic-embed-text-v2-moe.Q4_K_M.gguf",
}

LLAMACPP_MODELS = {
    **LLAMACPP_EMBED_MODELS
}

LLAMACPP_MODELS_REVERSED = {
    v: k for k, v in LLAMACPP_MODELS.items()
}

# Context sizes (max tokens) for each model
# Maximum context sizes for each model
LLAMACPP_MODEL_CONTEXTS = {
    "embeddinggemma": 2048,  # https://huggingface.co/google/embeddinggemma-300m
    "nomic-embed-text": 8192,  # https://huggingface.co/nomic-ai/nomic-embed-text-v1.5
    "nomic-embed-text-v2-moe": 2048,  # https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe
}

# Embedding sizes for each model
LLAMACPP_MODEL_EMBEDDING_SIZES = {
    "embeddinggemma": 768,
    "nomic-embed-text": 768,
    "nomic-embed-text-v2-moe": 768,
}

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
