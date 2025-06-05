from typing import Dict, Tuple, Optional, TypedDict, Union
from jet.models.base import scan_local_hf_models
from jet.transformers.formatters import format_json
from jet.utils.object import max_getattr
from jet.logger import logger
from transformers import AutoConfig
from jet.models.model_types import ModelKey, ModelType, ModelValue
from jet.models.constants import ALL_MODELS, MODEL_CONTEXTS, MODEL_EMBEDDING_TOKENS, MODEL_VALUES_LIST


def resolve_model_key(model: ModelType) -> ModelKey:
    """
    Retrieves the model key (short name) for a given model key or path.

    Args:
        model: A model key (short name) or full model path.

    Returns:
        The corresponding model key (short name).

    Raises:
        ValueError: If the model key or path is not recognized.
    """
    if model in ALL_MODELS:
        return model
    for key, value in ALL_MODELS.items():
        if value == model:
            return key
    raise ValueError(
        f"Invalid model: {model}. Must be one of: "
        f"{list(ALL_MODELS.keys()) + list(ALL_MODELS.values())}"
    )


def resolve_model_value(model: ModelType) -> ModelValue:
    """
    Retrieves the model value (full path) for a given model key or path.

    Args:
        model: A model key (short name) or full model path.

    Returns:
        The corresponding model value (full path).

    Raises:
        ValueError: If the model key or path is not recognized.
    """
    if model in ALL_MODELS:
        return ALL_MODELS[model]
    if model in ALL_MODELS.values():
        return model
    raise ValueError(
        f"Invalid model: {model}. Must be one of: "
        f"{list(ALL_MODELS.keys()) + list(ALL_MODELS.values())}"
    )


def get_model_limits(model_id: str | ModelValue) -> Tuple[Optional[int], Optional[int]]:
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

    max_context = max_getattr(config, 'max_position_embeddings', None)
    # or `config.hidden_dim`
    max_embeddings = max_getattr(config, 'hidden_size', None)

    return max_context, max_embeddings


class ModelInfoDict(TypedDict):
    models: Dict[ModelKey, ModelValue]
    contexts: Dict[ModelKey, int]
    embeddings: Dict[ModelKey, int]


def get_model_info() -> ModelInfoDict:
    model_info: ModelInfoDict = {
        "models": {},
        "contexts": {},
        "embeddings": {}
    }
    model_paths = scan_local_hf_models()
    for model_path in model_paths:
        if model_path not in MODEL_VALUES_LIST:
            logger.warning(f"Skipping unavailable model: {model_path}")
            continue

        try:
            short_name = resolve_model_key(model_path)
            max_contexts, max_embeddings = get_model_limits(model_path)
            if not max_contexts:
                raise ValueError(
                    f"Missing 'max_position_embeddings' from {model_path} config")
            elif not max_embeddings:
                raise ValueError(
                    f"Missing 'hidden_size' from {model_path} config")

            print(
                f"{short_name}: max_contexts={max_contexts}, max_embeddings={max_embeddings}")

            model_info["models"][short_name] = model_path
            model_info["contexts"][short_name] = max_contexts
            model_info["embeddings"][short_name] = max_embeddings

        except Exception as e:
            logger.error(
                f"Failed to get config for {short_name}: {e}", exc_info=True)
            continue

    return model_info


def resolve_model(model_name: ModelType) -> ModelType:
    """
    Resolves a model name or path against available models.

    Args:
        model_name: A short key or full model path.

    Returns:
        The resolved full model path.

    Raises:
        ValueError: If the model name/path is not recognized.
    """
    if model_name in ALL_MODELS:
        return ALL_MODELS[model_name]
    elif model_name in ALL_MODELS.values():
        return model_name
    else:
        raise ValueError(
            f"Invalid model: {model_name}. Must be one of: "
            f"{list(ALL_MODELS.keys()) + list(ALL_MODELS.values())}"
        )


def get_embedding_size(model: ModelType) -> int:
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
    if model_key not in MODEL_EMBEDDING_TOKENS:
        raise ValueError(f"Missing embedding size for model: {model_key}")
    return MODEL_EMBEDDING_TOKENS[model_key]


def get_context_size(model: ModelType) -> int:
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
    if model_key not in MODEL_CONTEXTS:
        raise ValueError(f"Missing context size for model: {model_key}")
    return MODEL_CONTEXTS[model_key]
