from typing import Dict, Tuple, Optional, TypedDict, Union
from jet.models.base import scan_local_hf_models
from jet.transformers.formatters import format_json
from jet.utils.object import max_getattr
from jet.logger import logger
from transformers import AutoConfig
from .mlx_types import (
    EmbedModelKey,
    EmbedModelValue,
    LLMModelKey,
    LLMModelValue,
    ModelKey,
    ModelType,
    ModelValue,
    model_keys_list,
    model_types_list,
    model_values_list,
)


MODEL_KEYS_LIST = list(set(model_keys_list))
MODEL_VALUES_LIST = list(set(model_values_list))
MODEL_TYPES_LIST = list(set(model_types_list))


AVAILABLE_LLM_MODELS: Dict[LLMModelKey, LLMModelValue] = {
    "dolphin3.0-llama3.1-8b-4bit": "mlx-community/Dolphin3.0-Llama3.1-8B-4bit",
    "llama-3.1-8b-instruct-4bit": "mlx-community/Llama-3.1-8B-Instruct-4bit",
    "llama-3.2-1b-instruct-4bit": "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "llama-3.2-3b-instruct-4bit": "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "mistral-nemo-instruct-2407-4bit": "mlx-community/Mistral-Nemo-Instruct-2407-4bit",
    "qwen2.5-7b-instruct-4bit": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "qwen2.5-14b-instruct-4bit": "mlx-community/Qwen2.5-14B-Instruct-4bit",
    "qwen2.5-coder-14b-instruct-4bit": "mlx-community/Qwen2.5-Coder-14B-Instruct-4bit",
    "qwen3-0.6b-4bit": "mlx-community/Qwen3-0.6B-4bit-DWQ",
    "qwen3-1.7b-4bit": "mlx-community/Qwen3-1.7B-4bit-DWQ",
    "qwen3-4b-4bit": "mlx-community/Qwen3-4B-4bit-DWQ",
    "qwen3-8b-4bit": "mlx-community/Qwen3-8B-4bit-DWQ",
    # Additional completions
    "deepseek-r1": "deepseek-ai/DeepSeek-R1",
    "deepseek-r1-distill": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "gemma-3-1b-it": "google/gemma-3-1b-it",
    "gemma-3-4b-it": "google/gemma-3-4b-it",
    "gemma-3-1b-it-qat-4bit": "mlx-community/gemma-3-1b-it-qat-4bit",
    "gemma-3-4b-it-qat-4bit": "mlx-community/gemma-3-4b-it-qat-4bit",
    "gemma-3-12b-it-qat-4bit": "mlx-community/gemma-3-12b-it-qat-4bit",
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B",
    "llama-3.2-1b": "meta-llama/Llama-3.2-1B",
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B",
    "qwen3-0.6b": "mlx-community/Qwen3-0.6B-4bit",
    "qwen3-1.7b": "mlx-community/Qwen3-1.7B-3bit",
    "qwen3-4b": "mlx-community/Qwen3-4B-3bit",
    "qwen1.5-0.5b-chat-4bit": "mlx-community/Qwen1.5-0.5B-Chat-4bit",
}


AVAILABLE_EMBED_MODELS: Dict[EmbedModelKey, EmbedModelValue] = {
    # HF
    "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
    "all-MiniLM-L12-v2": "sentence-transformers/all-MiniLM-L12-v2",
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "paraphrase-MiniLM-L12-v2": "sentence-transformers/paraphrase-MiniLM-L12-v2",
    "paraphrase-multilingual": "sentence-transformers/paraphrase-MiniLM-L12-v2",
    "bge-large": "BAAI/bge-large-en-v1.5",
    "nomic-embed-text": "nomic-ai/nomic-embed-text-v1.5",
    "mxbai-embed-large": "mixedbread-ai/mxbai-embed-large-v1",
    "granite-embedding": "ibm-granite/granite-embedding-30m-english",
    "granite-embedding:278m": "ibm-granite/granite-embedding-278m-multilingual",
    "specter": "allenai/specter",
    "e5-base-v2": "intfloat/e5-base-v2",
    "nomic-bert-2048": "nomic-ai/nomic-bert-2048",
    # Snowflake
    "snowflake-arctic-embed-s": "Snowflake/snowflake-arctic-embed-s",
    "snowflake-arctic-embed-m": "Snowflake/snowflake-arctic-embed-m",
    "snowflake-arctic-embed:137m": "Snowflake/snowflake-arctic-embed-m-long",
    # MLX
    "all-minilm-l6-v2-bf16": "mlx-community/all-MiniLM-L6-v2-bf16",
    "all-minilm-l6-v2-8bit": "mlx-community/all-MiniLM-L6-v2-8bit",
    "all-minilm-l6-v2-6bit": "mlx-community/all-MiniLM-L6-v2-6bit",
    "all-minilm-l6-v2-4bit": "mlx-community/all-MiniLM-L6-v2-4bit",
}

AVAILABLE_MODELS: Dict[ModelKey, ModelValue] = {
    **AVAILABLE_LLM_MODELS,
    **AVAILABLE_EMBED_MODELS,
}


ALL_MODELS: Dict[ModelKey, ModelValue] = {
    **AVAILABLE_LLM_MODELS,
    **AVAILABLE_EMBED_MODELS,
}

MODEL_CONTEXTS: Dict[ModelType, int] = {
    "dolphin3.0-llama3.1-8b-4bit": 131072,
    "llama-3.1-8b-instruct-4bit": 131072,
    "llama-3.2-1b-instruct-4bit": 131072,
    "llama-3.2-3b-instruct-4bit": 131072,
    "mistral-nemo-instruct-2407-4bit": 1024000,
    "qwen2.5-7b-instruct-4bit": 32768,
    "qwen2.5-14b-instruct-4bit": 32768,
    "qwen2.5-coder-14b-instruct-4bit": 32768,
    "deepseek-r1": 163840,
    "deepseek-r1-distill": 131072,
    "llama-3.1-8b": 131072,
    "llama-3.2-1b": 131072,
    "llama-3.2-3b": 131072,
    "qwen1.5-0.5b-chat-4bit": 32768,
    "all-mpnet-base-v2": 514,
    "all-MiniLM-L12-v2": 512,
    "all-MiniLM-L6-v2": 512,
    "paraphrase-MiniLM-L12-v2": 512,
    "paraphrase-multilingual": 512,
    "bge-large": 512,
    "nomic-embed-text": 8192,
    "mxbai-embed-large": 512,
    "granite-embedding": 514,
    "granite-embedding:278m": 514,
    "specter": 512,
    "e5-base-v2": 512,
    "nomic-bert-2048": 2048,
    "snowflake-arctic-embed-s": 512,
    "snowflake-arctic-embed-m": 512,
    "snowflake-arctic-embed:137m": 8192,
    "all-minilm-l6-v2-bf16": 512,
    "all-minilm-l6-v2-8bit": 512,
    "all-minilm-l6-v2-6bit": 512,
    "all-minilm-l6-v2-4bit": 512
}

MODEL_EMBEDDING_TOKENS: Dict[ModelType, int] = {
    "dolphin3.0-llama3.1-8b-4bit": 4096,
    "llama-3.1-8b-instruct-4bit": 4096,
    "llama-3.2-1b-instruct-4bit": 2048,
    "llama-3.2-3b-instruct-4bit": 3072,
    "mistral-nemo-instruct-2407-4bit": 5120,
    "qwen2.5-7b-instruct-4bit": 3584,
    "qwen2.5-14b-instruct-4bit": 5120,
    "qwen2.5-coder-14b-instruct-4bit": 5120,
    "deepseek-r1": 7168,
    "deepseek-r1-distill": 1536,
    "llama-3.1-8b": 4096,
    "llama-3.2-1b": 2048,
    "llama-3.2-3b": 3072,
    "qwen1.5-0.5b-chat-4bit": 1024,
    "all-mpnet-base-v2": 768,
    "all-MiniLM-L12-v2": 384,
    "all-MiniLM-L6-v2": 384,
    "paraphrase-MiniLM-L12-v2": 384,
    "paraphrase-multilingual": 384,
    "bge-large": 1024,
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
    "granite-embedding": 384,
    "granite-embedding:278m": 768,
    "specter": 768,
    "e5-base-v2": 768,
    "nomic-bert-2048": 768,
    "snowflake-arctic-embed-s": 384,
    "snowflake-arctic-embed-m": 768,
    "snowflake-arctic-embed:137m": 768,
    "all-minilm-l6-v2-bf16": 384,
    "all-minilm-l6-v2-8bit": 384,
    "all-minilm-l6-v2-6bit": 384,
    "all-minilm-l6-v2-4bit": 384
}


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
    model_info: ModelInfoDict = {"contexts": {}, "embeddings": {}}
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
