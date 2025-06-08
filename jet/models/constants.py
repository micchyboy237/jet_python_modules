from typing import Dict, List, Tuple, Optional, TypedDict, Union
from jet.models.base import scan_local_hf_models
from jet.transformers.formatters import format_json
from jet.utils.object import max_getattr
from jet.logger import logger
from transformers import AutoConfig
from jet.models.model_types import (
    EmbedModelKey,
    EmbedModelValue,
    LLMModelKey,
    LLMModelValue,
    ModelKey,
    ModelType,
    ModelValue,
    model_keys_list,
    model_values_list,
    model_types_list,
)


MODEL_KEYS_LIST: List[str] = model_keys_list
MODEL_VALUES_LIST: List[str] = model_values_list
MODEL_TYPES_LIST: List[str] = model_types_list


AVAILABLE_LLM_MODELS: Dict[LLMModelKey, LLMModelValue] = {
    # HF
    "dolphin3.0-llama3.1-8b-4bit": "mlx-community/Dolphin3.0-Llama3.1-8B-4bit",
    "llama-3.1-8b-instruct-4bit": "mlx-community/Llama-3.1-8B-Instruct-4bit",
    "llama-3.2-1b-instruct-4bit": "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "llama-3.2-3b-instruct-4bit": "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "mistral-nemo-instruct-2407-4bit": "mlx-community/Mistral-Nemo-Instruct-2407-4bit",
    "qwen2.5-7b-instruct-4bit": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "qwen2.5-14b-instruct-4bit": "mlx-community/Qwen2.5-14B-Instruct-4bit",
    "qwen2.5-coder-14b-instruct-4bit": "mlx-community/Qwen2.5-Coder-14B-Instruct-4bit",
    "qwen3-0.6b-4bit": "mlx-community/Qwen3-0.6B-4bit-DWQ",
    "qwen3-embedding-0.6b-4bit": "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
    "qwen3-1.7b-4bit": "mlx-community/Qwen3-1.7B-4bit-DWQ",
    "qwen3-4b-4bit": "mlx-community/Qwen3-4B-4bit-DWQ",
    "qwen3-8b-4bit": "mlx-community/Qwen3-8B-4bit-DWQ",
    "qwen3-embedding-0.6b": "Qwen/Qwen3-Embedding-0.6B",
    "qwen3-embedding-0.6b-gguf": "Qwen/Qwen3-Embedding-0.6B-GGUF",
    "qwen3-reranker-0.6b": "Qwen/Qwen3-Reranker-0.6B",
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
    "qwen3-embedding-0.6b-gguf": "Qwen/Qwen3-Embedding-0.6B-GGUF",
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
ALL_MODELS_REVERSED: Dict[ModelValue, ModelKey] = {
    v: k for k, v in ALL_MODELS.items()}
ALL_MODEL_KEYS: List[ModelKey] = list(ALL_MODELS.keys())
ALL_MODEL_VALUES: List[ModelValue] = list(ALL_MODELS.values())

MODEL_CONTEXTS: Dict[ModelType, int] = {
    "bge-large": 512,
    "qwen3-embedding-0.6b": 32768,
    "qwen3-reranker-0.6b": 40960,
    "snowflake-arctic-embed-m": 512,
    "snowflake-arctic-embed:137m": 8192,
    "snowflake-arctic-embed-s": 512,
    "granite-embedding:278m": 514,
    "granite-embedding": 514,
    "mxbai-embed-large": 512,
    "dolphin3.0-llama3.1-8b-4bit": 131072,
    "llama-3.1-8b-instruct-4bit": 131072,
    "llama-3.2-1b-instruct-4bit": 131072,
    "llama-3.2-3b-instruct-4bit": 131072,
    "mistral-nemo-instruct-2407-4bit": 1024000,
    "qwen2.5-14b-instruct-4bit": 32768,
    "qwen2.5-7b-instruct-4bit": 32768,
    "qwen2.5-coder-14b-instruct-4bit": 32768,
    "qwen3-4b-4bit": 40960,
    "qwen3-8b-4bit": 40960,
    "qwen3-embedding-0.6b-4bit": 32768,
    "all-minilm-l6-v2-4bit": 512,
    "all-minilm-l6-v2-6bit": 512,
    "all-minilm-l6-v2-8bit": 512,
    "all-minilm-l6-v2-bf16": 512,
    "nomic-embed-text": 8192,
    "all-MiniLM-L12-v2": 512,
    "all-MiniLM-L6-v2": 512,
    "all-mpnet-base-v2": 514,
    "paraphrase-multilingual": 512
}

MODEL_EMBEDDING_TOKENS: Dict[ModelType, int] = {
    "bge-large": 1024,
    "qwen3-embedding-0.6b": 1024,
    "qwen3-reranker-0.6b": 1024,
    "snowflake-arctic-embed-m": 768,
    "snowflake-arctic-embed:137m": 768,
    "snowflake-arctic-embed-s": 384,
    "granite-embedding:278m": 768,
    "granite-embedding": 384,
    "mxbai-embed-large": 1024,
    "dolphin3.0-llama3.1-8b-4bit": 4096,
    "llama-3.1-8b-instruct-4bit": 4096,
    "llama-3.2-1b-instruct-4bit": 2048,
    "llama-3.2-3b-instruct-4bit": 3072,
    "mistral-nemo-instruct-2407-4bit": 5120,
    "qwen2.5-14b-instruct-4bit": 5120,
    "qwen2.5-7b-instruct-4bit": 3584,
    "qwen2.5-coder-14b-instruct-4bit": 5120,
    "qwen3-4b-4bit": 2560,
    "qwen3-8b-4bit": 4096,
    "qwen3-embedding-0.6b-4bit": 1024,
    "all-minilm-l6-v2-4bit": 384,
    "all-minilm-l6-v2-6bit": 384,
    "all-minilm-l6-v2-8bit": 384,
    "all-minilm-l6-v2-bf16": 384,
    "nomic-embed-text": 768,
    "all-MiniLM-L12-v2": 384,
    "all-MiniLM-L6-v2": 384,
    "all-mpnet-base-v2": 768,
    "paraphrase-multilingual": 384
}
