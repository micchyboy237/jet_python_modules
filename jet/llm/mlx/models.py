# Configuration file for available MLX models with shortened names
from typing import Dict, Tuple, Optional, TypedDict
from jet.transformers.formatters import format_json
from jet.utils.object import max_getattr
from jet.logger import logger
from transformers import AutoConfig
from .mlx_types import ModelKey, ModelValue, ModelType

AVAILABLE_MODELS: Dict[ModelKey, ModelValue] = {
    "dolphin3.0-llama3.1-8b-4bit": "mlx-community/Dolphin3.0-Llama3.1-8B-4bit",
    "gemma-3-1b-it-qat-4bit": "mlx-community/gemma-3-1b-it-qat-4bit",
    "gemma-3-4b-it-qat-4bit": "mlx-community/gemma-3-4b-it-qat-4bit",
    "llama-3.1-8b-instruct-4bit": "mlx-community/Llama-3.1-8B-Instruct-4bit",
    "llama-3.2-1b-instruct-4bit": "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "llama-3.2-3b-instruct-4bit": "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "mistral-nemo-instruct-2407-4bit": "mlx-community/Mistral-Nemo-Instruct-2407-4bit",
    "qwen2.5-7b-instruct-4bit": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "qwen2.5-14b-instruct-4bit": "mlx-community/Qwen2.5-14B-Instruct-4bit",
    "qwen2.5-coder-14b-instruct-4bit": "mlx-community/Qwen2.5-Coder-14B-Instruct-4bit",
    "qwen3-0.6b-4bit": "mlx-community/Qwen3-0.6B-4bit",
    "qwen3-1.7b-3bit": "mlx-community/Qwen3-1.7B-3bit",
    "qwen3-4b-3bit": "mlx-community/Qwen3-4B-3bit",
    "qwen3-8b-3bit": "mlx-community/Qwen3-8B-3bit",
}

MODEL_CONTEXTS: Dict[ModelKey, int] = {
    "dolphin3.0-llama3.1-8b-4bit": 131072,
    "gemma-3-1b-it-qat-4bit": 32768,
    "gemma-3-4b-it-qat-4bit": 131072,
    "llama-3.1-8b-instruct-4bit": 131072,
    "llama-3.2-1b-instruct-4bit": 131072,
    "llama-3.2-3b-instruct-4bit": 131072,
    "mistral-nemo-instruct-2407-4bit": 1024000,
    "qwen2.5-7b-instruct-4bit": 32768,
    "qwen2.5-14b-instruct-4bit": 32768,
    "qwen2.5-coder-14b-instruct-4bit": 32768,
    "qwen3-0.6b-4bit": 40960,
    "qwen3-1.7b-3bit": 40960,
    "qwen3-4b-3bit": 40960,
    "qwen3-8b-3bit": 40960
}

MODEL_EMBEDDING_TOKENS: Dict[ModelKey, int] = {
    "dolphin3.0-llama3.1-8b-4bit": 4096,
    "gemma-3-1b-it-qat-4bit": 1152,
    "gemma-3-4b-it-qat-4bit": 2560,
    "llama-3.1-8b-instruct-4bit": 4096,
    "llama-3.2-1b-instruct-4bit": 2048,
    "llama-3.2-3b-instruct-4bit": 3072,
    "mistral-nemo-instruct-2407-4bit": 5120,
    "qwen2.5-7b-instruct-4bit": 3584,
    "qwen2.5-14b-instruct-4bit": 5120,
    "qwen2.5-coder-14b-instruct-4bit": 5120,
    "qwen3-0.6b-4bit": 1024,
    "qwen3-1.7b-3bit": 2048,
    "qwen3-4b-3bit": 2560,
    "qwen3-8b-3bit": 4096
}


def get_model_limits(model_id: ModelType) -> Tuple[Optional[int], Optional[int]]:
    config = AutoConfig.from_pretrained(model_id)

    max_context = max_getattr(config, 'max_position_embeddings', None)
    # or `config.hidden_dim`
    max_embeddings = max_getattr(config, 'hidden_size', None)

    return max_context, max_embeddings


class ModelInfoDict(TypedDict):
    contexts: Dict[ModelKey, int]
    embeddings: Dict[ModelKey, int]


def get_model_info() -> ModelInfoDict:
    model_info: ModelInfoDict = {"contexts": {}, "embeddings": {}}
    for short_name, model_path in AVAILABLE_MODELS.items():
        try:
            max_contexts, max_embeddings = get_model_limits(model_path)
            if not max_contexts:
                raise ValueError(
                    f"Missing 'max_position_embeddings' from {model_path} config")
            elif not max_embeddings:
                raise ValueError(
                    f"Missing 'hidden_size' from {model_path} config")

            print(
                f"{short_name}: max_contexts={max_contexts}, max_embeddings={max_embeddings}")

            model_info["contexts"][short_name] = max_contexts
            model_info["embeddings"][short_name] = max_embeddings

        except Exception as e:
            logger.error(
                f"Failed to get config for {short_name}: {e}", exc_info=True)
            raise

    return model_info


def resolve_model(model_name: ModelType, available_models: dict) -> ModelType:
    """
    Resolves a model name or path against available models.

    Args:
        model_name: A short key or full model path.
        available_models: A dictionary of {short_name: full_path}.

    Returns:
        The resolved full model path.

    Raises:
        ValueError: If the model name/path is not recognized.
    """
    if model_name in available_models:
        return available_models[model_name]
    elif model_name in available_models.values():
        return model_name
    else:
        raise ValueError(
            f"Invalid model: {model_name}. Must be one of: "
            f"{list(available_models.keys()) + list(available_models.values())}"
        )
