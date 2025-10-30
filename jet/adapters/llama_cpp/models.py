LLAMACPP_LLM_MODELS = {
    "llama-3.1:8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama-3.2-instruct:3b": "meta-llama/Llama-3.2-3B-Instruct",
    "qwen3-instruct-2507:4b": "Qwen/Qwen3-4B-Instruct-2507",
}

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
    **LLAMACPP_LLM_MODELS,
    **LLAMACPP_EMBED_MODELS,
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
    "qwen3-instruct-2507:4b": 262144, # https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507
    "llama-3.1:8b": 131072, # https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
    "llama-3.2-instruct:3b": 131072, # https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
}

# Embedding sizes for each model
LLAMACPP_MODEL_EMBEDDING_SIZES = {
    "embeddinggemma": 768,
    "nomic-embed-text": 768,
    "nomic-embed-text-v2-moe": 768,
    "qwen3-instruct-2507:4b": 2560,
    "llama-3.1:8b": 4096,
    "llama-3.2-instruct:3b": 3072,
}
