from typing import Literal, TypedDict, Union

import numpy as np

LLAMACPP_LLM_KEYS = Literal[
    "smollm3:3b",
    "llama-3.2:3b",
    "gemma-3:4b",
    "qwen3:4b",
    "qwen2.5:7b",
    "deepseek-r1:1.5b-q5km",
    "deepseek-r1:1.5b-q5kl",
    "mistral-nemo:12b-ish",
    "deepseek-r1:7b",
    "llama-3.1:8b",
    "qwen3.5:0.8b",
    "qwen3.5:2b",
    "qwen3.5:4b",
    "ministral:3b",
    "deepseek-coder-v2-lite:16b-ish",
    "lfm2-enjp:350m",
    "gemma-2-jpn-translate:2b",
    "shisa-llama3.2:3b-q4",
    "shisa-llama3.2:3b-iq4",
    "shisa-lfm2:1.2b",
    "sarashina:3b",
    "elyza-jp:8b-iq2",
    "alma-ja:7b",
    "nano-imp:1b-q8",
    "dolphin-2.6-phi:2b",
    "fiendish-llama:3b",
    "llama-3.2-uncensored:3b",
    "impish-llama:4b",
    "wizardlm-uncensored:7b",
    "gemma3-uncensored:1b",
    "qwen3.5-uncensored:2b",
    "qwen3.5-uncensored:4b",
]

LLAMACPP_LLM_VALUES = Literal[
    "HuggingFaceTB/SmolLM3-3B",
    "meta-llama/Llama-3.2-3B-Instruct",
    "google/gemma-3-4b-it",
    "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen/Qwen2.5-7B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "mistralai/Mistral-Nemo-Instruct-2407",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen3.5-0.8B",
    "Qwen/Qwen3.5-2B",
    "Qwen/Qwen3.5-4B",
    "ministral/Ministral-3b-instruct",
    "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    "LiquidAI/LFM2-350M-ENJP-MT",
    "webbigdata/gemma-2-2b-jpn-it-translate",
    "shisa-ai/shisa-v2.1-llama3.2-3b",
    "shisa-ai/shisa-v2.1-llama3.2-3b",
    "shisa-ai/shisa-v2.1-lfm2-1.2b",
    "sbintuitions/sarashina2.2-3b-instruct-v0.1",
    "elyza/Llama-3-ELYZA-JP-8B",
    "webbigdata/ALMA-7B-Ja-V2",
    "SicariusSicariiStuff/Nano_Imp_1B",
    "cognitivecomputations/dolphin-2.6-phi-2",
    "SicariusSicariiStuff/Fiendish_LLAMA_3B",
    "chuanli11/Llama-3.2-3B-Instruct-uncensored",
    "SicariusSicariiStuff/Impish_LLAMA_4B",
    "ehartford/WizardLM-7B-Uncensored",
    "SicariusSicariiStuff/Gemma3-UNCENSORED-1B",
    "HauhauCS/Qwen3.5-2B-Uncensored-HauhauCS-Aggressive",
    "HauhauCS/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive",
]

LLAMACPP_EMBED_KEYS = Literal[
    "nomic-embed:1.5",
    "nomic-embed:2-moe",
    "all-minilm:l12-q4",
    "embedding-gemma:300m",
    "qwen3-embed:4b-q5_0",
    "qwen3-embed:0.6b",
]

LLAMACPP_EMBED_VALUES = Literal[
    "nomic-ai/nomic-embed-text-v1.5",
    "nomic-ai/nomic-embed-text-v2-moe",
    "sentence-transformers/all-MiniLM-L12-v2",
    "google/embeddinggemma-300m",
    "Qwen/Qwen3-Embedding-4B",
    "Qwen/Qwen3-Embedding-0.6B",
]

LLAMACPP_RERANK_KEYS = Literal[
    "bge-rerank:v2-m3",
    "bge-rerank:large",
    "qwen3-rerank:4b",
    "qwen3-rerank:0.6b",
]

LLAMACPP_RERANK_VALUES = Literal[
    "BAAI/bge-reranker-v2-m3",
    "BAAI/bge-reranker-large",
    "Qwen/Qwen3-Reranker-4B",
    "Qwen/Qwen3-Reranker-0.6B",
]

LLAMACPP_VISION_KEYS = Literal[
    "gemma-3-vision:4b",
    "qwen2.5-vl:7b",
    "qwen3.5:0.8b-vision",
    "qwen3.5:2b-vision",
    "qwen3.5:4b-vision",
]

LLAMACPP_VISION_VALUES = Literal[
    "google/gemma-3-4b-it",
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen/Qwen3.5-0.8B",
    "Qwen/Qwen3.5-2B",
    "Qwen/Qwen3.5-4B",
]

# Composite types
LLAMACPP_LLM_TYPES = LLAMACPP_LLM_KEYS | LLAMACPP_LLM_VALUES
LLAMACPP_EMBED_TYPES = LLAMACPP_EMBED_KEYS | LLAMACPP_EMBED_VALUES
LLAMACPP_RERANK_TYPES = LLAMACPP_RERANK_KEYS | LLAMACPP_RERANK_VALUES
LLAMACPP_VISION_TYPES = LLAMACPP_VISION_KEYS | LLAMACPP_VISION_VALUES
LLAMACPP_KEYS = (
    LLAMACPP_LLM_KEYS
    | LLAMACPP_EMBED_KEYS
    | LLAMACPP_RERANK_KEYS
    | LLAMACPP_VISION_KEYS
)
LLAMACPP_VALUES = (
    LLAMACPP_LLM_VALUES
    | LLAMACPP_EMBED_VALUES
    | LLAMACPP_RERANK_VALUES
    | LLAMACPP_VISION_VALUES
)
LLAMACPP_TYPES = LLAMACPP_KEYS | LLAMACPP_VALUES


# ────────────────────────────────────────────────────────────────────────────────
# Embedding Type Definitions
# ────────────────────────────────────────────────────────────────────────────────

EmbeddingVector = list[float] | np.ndarray
EmbeddingBatch = list[EmbeddingVector]
EmbeddingOutput = Union[list[float], list[list[float]], np.ndarray]
EmbeddingInput = str | list[str]
EmbeddingInputType = Literal["query", "document", "default"]


class EmbeddingResultItem(TypedDict):
    object: Literal["embedding"]
    embedding: list[float]
    index: int


class EmbeddingResponse(TypedDict):
    object: Literal["list"]
    data: list[EmbeddingResultItem]
    model: str
    usage: dict[str, int]


IDType = str
MetadataType = dict


class SearchResultType(TypedDict):
    rank: int | None
    index: int
    text: str
    score: float
    id: IDType | None  # optional — only present when provided
    metadata: MetadataType | None  # optional — only present when provided


GenerateEmbeddingsReturnType = EmbeddingOutput
