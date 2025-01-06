# LLM and embedding config

base_url = "http://localhost:11434"
base_embed_url = "http://localhost:11434"
small_llm_model = "llama3.2"
large_llm_model = "llama3.1"
small_embed_model = "mxbai-embed-large"
large_embed_model = "nomic-embed-text"
DEFAULT_LLM_SETTINGS = {
    "model": large_llm_model,
    "context_window": 4096,
    "request_timeout": 300.0,
    "temperature": 0,
    "base_url": base_url,
}
DEFAULT_EMBED_SETTINGS = {
    "model_name": large_embed_model,
    "base_url": base_embed_url,
    "embed_batch_size": 32,
    "ollama_additional_kwargs": {}
}
