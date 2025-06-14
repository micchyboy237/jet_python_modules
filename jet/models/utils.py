import os
from typing import Dict, List, Tuple, Optional, TypedDict, Union
from jet.models.base import scan_local_hf_models
from jet.models.onnx_model_checker import has_onnx_model_in_repo
from jet.transformers.formatters import format_json
from jet.utils.object import max_getattr
from jet.logger import logger
from transformers import AutoConfig
from jet.models.model_types import ModelKey, ModelType, ModelValue
from jet.models.constants import ALL_MODEL_VALUES, ALL_MODELS, ALL_MODELS_REVERSED, MODEL_CONTEXTS, MODEL_EMBEDDING_TOKENS, MODEL_VALUES_LIST


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
    elif model in ALL_MODEL_VALUES:
        return ALL_MODELS_REVERSED[model]
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


def generate_short_name(model_id: str) -> Optional[str]:
    """
    Extract the final segment from a model ID or path and convert it to lowercase.

    Args:
        model_id (str): Model identifier or path (e.g., "mlx-community/Dolphin3.0-Llama3.1-8B-4bit")

    Returns:
        Optional[str]: Lowercase short name (e.g., "dolphin3.0-llama3.1-8b-4bit") or None if input is empty
    """
    try:
        if not model_id:
            logger.warning("Empty model_id provided")
            return None

        # Get the final path component and convert to lowercase
        short_name = os.path.basename(model_id).lower()
        logger.debug(f"Processed short name: {short_name}")
        return short_name
    except Exception as e:
        logger.error(f"Error processing model_id ({model_id}): {str(e)}")
        return None


class ModelInfoDict(TypedDict):
    models: Dict[ModelKey, ModelValue]
    contexts: Dict[ModelKey, int]
    embeddings: Dict[ModelKey, int]
    has_onnx: Dict[ModelKey, bool]
    missing: List[str]


def get_model_info() -> ModelInfoDict:
    model_info: ModelInfoDict = {
        "models": {},
        "contexts": {},
        "embeddings": {},
        "has_onnx": {},
        "missing": []
    }
    model_paths = scan_local_hf_models()
    for model_path in model_paths:
        try:
            if model_path not in MODEL_VALUES_LIST:
                short_name = generate_short_name(model_path)
            else:
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
            model_info["has_onnx"][short_name] = has_onnx_model_in_repo(
                model_path)

        except Exception as e:
            logger.error(
                f"Failed to get config for {short_name}: {str(e)[:100]}", exc_info=True)
            model_info["missing"].append(model_path)
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
