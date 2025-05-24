from typing import Dict, Tuple, Optional, TypedDict, Union
from jet.transformers.formatters import format_json
from jet.utils.object import max_getattr
from jet.logger import logger
from transformers import AutoConfig
from .mlx_types import LLMModelType, EmbedModelKey, EmbedModelType, EmbedModelValue, ModelKey, ModelType, ModelValue, LLMModelType

AVAILABLE_MODELS: Dict[ModelKey, ModelValue] = {
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
}

AVAILABLE_EMBED_MODELS: Dict[EmbedModelKey, EmbedModelValue] = {
    # HF
    "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
    "all-MiniLM-L12-v2": "sentence-transformers/all-MiniLM-L12-v2",
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    # Ollama
    "nomic-embed-text": "nomic-ai/nomic-embed-text-v1.5",
    "mxbai-embed-large": "mixedbread-ai/mxbai-embed-large-v1",
    "granite-embedding": "ibm-granite/granite-embedding-30m-english",
    "granite-embedding:278m": "ibm-granite/granite-embedding-278m-multilingual",
    "all-minilm:22m": "sentence-transformers/all-MiniLM-L6-v2",
    "all-minilm:33m": "sentence-transformers/all-MiniLM-L12-v2",
    "snowflake-arctic-embed:33m": "Snowflake/snowflake-arctic-embed-s",
    "snowflake-arctic-embed:137m": "Snowflake/snowflake-arctic-embed-m-long",
    "snowflake-arctic-embed": "Snowflake/snowflake-arctic-embed-l",
    "paraphrase-multilingual": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "bge-large": "BAAI/bge-large-en-v1.5",
    # MLX
    "all-minilm-l6-v2-bf16": "mlx-community/all-MiniLM-L6-v2-bf16",
    "all-minilm-l6-v2-8bit": "mlx-community/all-MiniLM-L6-v2-8bit",
    "all-minilm-l6-v2-6bit": "mlx-community/all-MiniLM-L6-v2-6bit",
    "all-minilm-l6-v2-4bit": "mlx-community/all-MiniLM-L6-v2-4bit",
}

ALL_MODELS: Dict[ModelType, Union[ModelValue, EmbedModelValue]] = {
    **AVAILABLE_MODELS,
    **AVAILABLE_EMBED_MODELS,
}

MODEL_CONTEXTS: Dict[LLMModelType, int] = {
    "dolphin3.0-llama3.1-8b-4bit": 131072,
    "llama-3.1-8b-instruct-4bit": 131072,
    "llama-3.2-1b-instruct-4bit": 131072,
    "llama-3.2-3b-instruct-4bit": 131072,
    "mistral-nemo-instruct-2407-4bit": 1024000,
    "qwen2.5-7b-instruct-4bit": 32768,
    "qwen2.5-14b-instruct-4bit": 32768,
    "qwen2.5-coder-14b-instruct-4bit": 32768,
    "qwen3-0.6b-4bit": 40960,
    "qwen3-1.7b-4bit": 40960,
    "qwen3-4b-4bit": 40960,
    "qwen3-8b-4bit": 40960,
    "nomic-embed-text": 8192,
    "mxbai-embed-large": 512,
    "granite-embedding": 514,
    "granite-embedding:278m": 514,
    "all-minilm:22m": 512,
    "all-minilm:33m": 512,
    "snowflake-arctic-embed:33m": 512,
    "snowflake-arctic-embed:137m": 8192,
    "snowflake-arctic-embed": 512,
    "paraphrase-multilingual": 514,
    "bge-large": 512,
    "all-minilm-l6-v2-bf16": 512,
    "all-minilm-l6-v2-8bit": 512,
    "all-minilm-l6-v2-6bit": 512,
    "all-minilm-l6-v2-4bit": 512
}

MODEL_EMBEDDING_TOKENS: Dict[LLMModelType, int] = {
    "dolphin3.0-llama3.1-8b-4bit": 4096,
    "llama-3.1-8b-instruct-4bit": 4096,
    "llama-3.2-1b-instruct-4bit": 2048,
    "llama-3.2-3b-instruct-4bit": 3072,
    "mistral-nemo-instruct-2407-4bit": 5120,
    "qwen2.5-7b-instruct-4bit": 3584,
    "qwen2.5-14b-instruct-4bit": 5120,
    "qwen2.5-coder-14b-instruct-4bit": 5120,
    "qwen3-0.6b-4bit": 1024,
    "qwen3-1.7b-4bit": 2048,
    "qwen3-4b-4bit": 2560,
    "qwen3-8b-4bit": 4096,
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
    "granite-embedding": 384,
    "granite-embedding:278m": 768,
    "all-minilm:22m": 384,
    "all-minilm:33m": 384,
    "snowflake-arctic-embed:33m": 384,
    "snowflake-arctic-embed:137m": 768,
    "snowflake-arctic-embed": 1024,
    "paraphrase-multilingual": 768,
    "bge-large": 1024,
    "all-minilm-l6-v2-bf16": 384,
    "all-minilm-l6-v2-8bit": 384,
    "all-minilm-l6-v2-6bit": 384,
    "all-minilm-l6-v2-4bit": 384,
    "all-mpnet-base-v2": 768,
    "all-MiniLM-L12-v2": 384,
    "all-MiniLM-L6-v2": 384,
}


def resolve_model_key(model: ModelType) -> Union[ModelKey, EmbedModelKey]:
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


def resolve_model_value(model: ModelType) -> Union[ModelValue, EmbedModelValue]:
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


def get_model_limits(model_id: LLMModelType) -> Tuple[Optional[int], Optional[int]]:
    config = AutoConfig.from_pretrained(model_id)

    max_context = max_getattr(config, 'max_position_embeddings', None)
    # or `config.hidden_dim`
    max_embeddings = max_getattr(config, 'hidden_size', None)

    return max_context, max_embeddings


class ModelInfoDict(TypedDict):
    contexts: Dict[LLMModelType, int]
    embeddings: Dict[LLMModelType, int]


def get_model_info() -> ModelInfoDict:
    model_info: ModelInfoDict = {"contexts": {}, "embeddings": {}}
    for short_name, model_path in ALL_MODELS.items():
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
