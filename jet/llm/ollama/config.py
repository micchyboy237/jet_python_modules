# LLM and embedding config

from jet.llm.ollama.constants import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_EMBED_BATCH_SIZE,
    DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_TEMPERATURE,
    OLLAMA_BASE_URL,
    OLLAMA_BASE_EMBED_URL,
    OLLAMA_LARGE_CHUNK_OVERLAP,
    OLLAMA_LARGE_CHUNK_SIZE,
    OLLAMA_SMALL_LLM_MODEL,
    OLLAMA_LARGE_LLM_MODEL,
    OLLAMA_SMALL_EMBED_MODEL,
    OLLAMA_LARGE_EMBED_MODEL,
)


base_url = OLLAMA_BASE_URL
base_embed_url = OLLAMA_BASE_EMBED_URL
small_llm_model = OLLAMA_SMALL_LLM_MODEL
large_llm_model = OLLAMA_LARGE_LLM_MODEL
small_embed_model = OLLAMA_SMALL_EMBED_MODEL
large_embed_model = OLLAMA_LARGE_EMBED_MODEL

DEFAULT_CHUNK_SIZE = OLLAMA_LARGE_CHUNK_SIZE  # tokens
DEFAULT_CHUNK_OVERLAP = OLLAMA_LARGE_CHUNK_OVERLAP  # tokens

DEFAULT_LLM_SETTINGS = {
    "model": large_llm_model,
    "context_window": DEFAULT_CONTEXT_WINDOW,
    "request_timeout": DEFAULT_REQUEST_TIMEOUT,
    "temperature": DEFAULT_TEMPERATURE,
    "base_url": base_url,
}
DEFAULT_EMBED_SETTINGS = {
    "model_name": large_embed_model,
    "base_url": base_embed_url,
    "embed_batch_size": DEFAULT_EMBED_BATCH_SIZE,
    "ollama_additional_kwargs": {}
}
