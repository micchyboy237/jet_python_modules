from jet.adapters.llama_cpp.types import (
    LLAMACPP_EMBED_KEYS,
    LLAMACPP_EMBED_VALUES,
    LLAMACPP_KEYS,
    LLAMACPP_LLM_KEYS,
    LLAMACPP_LLM_VALUES,
    LLAMACPP_RERANK_KEYS,
    LLAMACPP_RERANK_VALUES,
    LLAMACPP_VALUES,
    LLAMACPP_VISION_KEYS,
    LLAMACPP_VISION_VALUES,
)

LLAMACPP_LLM_MODELS: dict[LLAMACPP_LLM_KEYS, LLAMACPP_LLM_VALUES] = {
    "smollm3:3b": "HuggingFaceTB/SmolLM3-3B",
    "llama-3.2:3b": "meta-llama/Llama-3.2-3B-Instruct",
    "gemma-3:4b": "google/gemma-3-4b-it",
    "qwen3:4b": "Qwen/Qwen3-4B-Instruct-2507",
    "qwen2.5:7b": "Qwen/Qwen2.5-7B-Instruct",
    "deepseek-r1:1.5b-q5km": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "deepseek-r1:1.5b-q5kl": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "mistral-nemo:12b-ish": "mistralai/Mistral-Nemo-Instruct-2407",
    "deepseek-r1:7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "llama-3.1:8b": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen3.5:0.8b": "Qwen/Qwen3.5-0.8B",
    "qwen3.5:2b": "Qwen/Qwen3.5-2B",
    "qwen3.5:4b": "Qwen/Qwen3.5-4B",
    "ministral:3b": "ministral/Ministral-3b-instruct",
    "deepseek-coder-v2-lite:16b-ish": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    "lfm2-enjp:350m": "LiquidAI/LFM2-350M-ENJP-MT",
    "gemma-2-jpn-translate:2b": "webbigdata/gemma-2-2b-jpn-it-translate",
    "shisa-llama3.2:3b-q4": "shisa-ai/shisa-v2.1-llama3.2-3b",
    "shisa-llama3.2:3b-iq4": "shisa-ai/shisa-v2.1-llama3.2-3b",
    "shisa-lfm2:1.2b": "shisa-ai/shisa-v2.1-lfm2-1.2b",
    "sarashina:3b": "sbintuitions/sarashina2.2-3b-instruct-v0.1",
    "elyza-jp:8b-iq2": "elyza/Llama-3-ELYZA-JP-8B",
    "alma-ja:7b": "webbigdata/ALMA-7B-Ja-V2",
    "nano-imp:1b-q8": "SicariusSicariiStuff/Nano_Imp_1B",
    "dolphin-2.6-phi:2b": "cognitivecomputations/dolphin-2.6-phi-2",
    "fiendish-llama:3b": "SicariusSicariiStuff/Fiendish_LLAMA_3B",
    "llama-3.2-uncensored:3b": "chuanli11/Llama-3.2-3B-Instruct-uncensored",
    "impish-llama:4b": "SicariusSicariiStuff/Impish_LLAMA_4B",
    "wizardlm-uncensored:7b": "ehartford/WizardLM-7B-Uncensored",
    "gemma3-uncensored:1b": "SicariusSicariiStuff/Gemma3-UNCENSORED-1B",
    "qwen3.5-uncensored:2b": "Qwen/Qwen3.5-2B",
    "qwen3.5-uncensored:4b": "Qwen/Qwen3.5-4B",
}

LLAMACPP_EMBED_MODELS: dict[LLAMACPP_EMBED_KEYS, LLAMACPP_EMBED_VALUES] = {
    "nomic-embed:1.5": "nomic-ai/nomic-embed-text-v1.5",
    "nomic-embed:2-moe": "nomic-ai/nomic-embed-text-v2-moe",
    "all-minilm:l12-q4": "sentence-transformers/all-MiniLM-L12-v2",
    "embedding-gemma:300m": "google/embeddinggemma-300m",
    "qwen3-embed:4b-q5_0": "Qwen/Qwen3-Embedding-4B",
    "qwen3-embed:0.6b": "Qwen/Qwen3-Embedding-0.6B",
}

LLAMACPP_EMBED_MODELS_GGUF_MAPPING: dict[LLAMACPP_EMBED_KEYS, str] = {
    "nomic-embed:1.5": "nomic-embed-text-v1.5.Q4_K_M.gguf",
    "nomic-embed:2-moe": "nomic-embed-text-v2-moe.Q4_K_M.gguf",
    "all-minilm:l12-q4": "all-MiniLM-L12-v2-q4_0.gguf",
    "embedding-gemma:300m": "embeddinggemma-300M-Q8_0.gguf",
    "qwen3-embed:4b-q5_0": "Qwen3-Embedding-4B-Q5_0.gguf",
    "qwen3-embed:0.6b": "Qwen3-Embedding-0.6B-Q8_0.gguf",
}

LLAMACPP_RERANK_MODELS: dict[LLAMACPP_RERANK_KEYS, LLAMACPP_RERANK_VALUES] = {
    "bge-rerank:v2-m3": "BAAI/bge-reranker-v2-m3",
    "bge-rerank:large": "BAAI/bge-reranker-large",
    "qwen3-rerank:4b": "Qwen/Qwen3-Reranker-4B",
    "qwen3-rerank:0.6b": "Qwen/Qwen3-Reranker-0.6B",
}

# Vision models: alias -> (HF repo ID, relative GGUF path)
LLAMACPP_VISION_MODELS_MAPPING: dict[
    LLAMACPP_VISION_KEYS, tuple[LLAMACPP_VISION_VALUES, str]
] = {
    "gemma-3-vision:4b": (
        "google/gemma-3-4b-it",
        "ggml-org_gemma-3-4b-it-GGUF_gemma-3-4b-it-Q4_K_M.gguf",
    ),
    "qwen2.5-vl:7b": (
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "ggml-org_Qwen2.5-VL-7B-Instruct-GGUF_Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
    ),
    "qwen3.5:0.8b-vision": ("Qwen/Qwen3.5-0.8B", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    "qwen3.5:2b-vision": ("Qwen/Qwen3.5-2B", "Qwen3.5-2B-Q4_K_M.gguf"),
    "qwen3.5:4b-vision": ("Qwen/Qwen3.5-4B", "Qwen3.5-4B-Q4_K_M.gguf"),
}

LLAMACPP_MODELS: dict[LLAMACPP_KEYS, LLAMACPP_VALUES] = {
    **LLAMACPP_LLM_MODELS,  # type: ignore[dict-item]
    **LLAMACPP_EMBED_MODELS,  # type: ignore[dict-item]
    **LLAMACPP_RERANK_MODELS,  # type: ignore[dict-item]
    **{k: v[0] for k, v in LLAMACPP_VISION_MODELS_MAPPING.items()},  # type: ignore[dict-item]
}

LLAMACPP_MODELS_REVERSED: dict[str, LLAMACPP_KEYS] = {
    v: k for k, v in LLAMACPP_MODELS.items()
}

# Context windows and embedding sizes remain, updated with new keys:
LLAMACPP_MODEL_CONTEXTS: dict[LLAMACPP_KEYS, int] = {
    # LLM models (from models.llm.ini - context window `c`)
    "smollm3:3b": 4096,
    "llama-3.2:3b": 4096,
    "gemma-3:4b": 4096,
    "qwen3:4b": 4096,
    "qwen2.5:7b": 4096,
    "deepseek-r1:1.5b-q5km": 4096,
    "deepseek-r1:1.5b-q5kl": 4096,
    "mistral-nemo:12b-ish": 4096,
    "deepseek-r1:7b": 4096,
    "llama-3.1:8b": 4096,
    "qwen3.5:0.8b": 4096,
    "qwen3.5:2b": 4096,
    "qwen3.5:4b": 4096,
    "ministral:3b": 4096,
    "deepseek-coder-v2-lite:16b-ish": 4096,
    # Translators
    "lfm2-enjp:350m": 2048,
    "gemma-2-jpn-translate:2b": 2048,
    "shisa-llama3.2:3b-q4": 4096,
    "shisa-llama3.2:3b-iq4": 4096,
    "shisa-lfm2:1.2b": 4096,
    "sarashina:3b": 4096,
    "elyza-jp:8b-iq2": 4096,
    "alma-ja:7b": 4096,
    # Uncensored / Spicy
    "nano-imp:1b-q8": 4096,
    "dolphin-2.6-phi:2b": 4096,
    "fiendish-llama:3b": 4096,
    "llama-3.2-uncensored:3b": 4096,
    "impish-llama:4b": 4096,
    "wizardlm-uncensored:7b": 4096,
    "gemma3-uncensored:1b": 4096,
    "qwen3.5-uncensored:2b": 16384,
    "qwen3.5-uncensored:4b": 10000,
    # Embedding models (from models.embedders.ini - based on ubatch-size)
    "nomic-embed:1.5": 1024,
    "nomic-embed:2-moe": 512,
    "all-minilm:l12-q4": 256,
    "embedding-gemma:300m": 512,
    "qwen3-embed:4b-q5_0": 128,
    "qwen3-embed:0.6b": 512,
    # Reranker models (from models.rerankers.ini)
    "bge-rerank:v2-m3": 1024,
    "bge-rerank:large": 512,
    "qwen3-rerank:4b": 2048,
    "qwen3-rerank:0.6b": 2048,
    # Vision models (from models.llm.ini - context window `c`)
    "gemma-3-vision:4b": 4096,
    "qwen2.5-vl:7b": 4096,
    "qwen3.5:0.8b-vision": 4096,
    "qwen3.5:2b-vision": 4096,
    "qwen3.5:4b-vision": 4096,
}

# Note: Embedding sizes for non-embedding models (LLM, reranker, vision)
# are set to 0 or omitted. You may want to populate known values.
LLAMACPP_MODEL_EMBEDDING_SIZES: dict[LLAMACPP_KEYS, int] = {
    # Embedding models
    "nomic-embed:1.5": 768,
    "nomic-embed:2-moe": 768,
    "all-minilm:l12-q4": 384,
    "embedding-gemma:300m": 768,
    "qwen3-embed:4b-q5_0": 2560,
    "qwen3-embed:0.6b": 1024,
    # Reranker models (can also produce embeddings)
    "qwen3-rerank:4b": 2560,
    "qwen3-rerank:0.6b": 1024,
}
