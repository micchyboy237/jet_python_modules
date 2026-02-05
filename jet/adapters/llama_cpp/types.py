from typing import Literal, TypedDict

import numpy as np

# LLM model keys and values
LLAMACPP_LLM_KEYS = Literal[
    "llama-3.1:8b",
    "llama-3.2-instruct:3b",
    "qwen3-instruct-2507:4b",
]
LLAMACPP_LLM_VALUES = Literal[
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen3-4B-Instruct-2507",
]

# Embedding model keys and values
LLAMACPP_EMBED_KEYS = Literal[
    "embeddinggemma",
    "nomic-embed-text",
    "nomic-embed-text-v2-moe",
]
LLAMACPP_EMBED_VALUES = Literal[
    "google/embeddinggemma-300m",
    "nomic-ai/nomic-embed-text-v1.5",
    "nomic-ai/nomic-embed-text-v2-moe",
]

# LLM models
LLAMACPP_LLM_TYPES = LLAMACPP_LLM_KEYS | LLAMACPP_LLM_VALUES

# Embedding models
LLAMACPP_EMBED_TYPES = LLAMACPP_EMBED_KEYS | LLAMACPP_EMBED_VALUES

# Combined models (all keys and all values)
LLAMACPP_KEYS = LLAMACPP_LLM_KEYS | LLAMACPP_EMBED_KEYS
LLAMACPP_VALUES = LLAMACPP_LLM_VALUES | LLAMACPP_EMBED_VALUES
LLAMACPP_TYPES = LLAMACPP_KEYS | LLAMACPP_VALUES


# ────────────────────────────────────────────────────────────────────────────────
# Embedding Type Definitions
# ────────────────────────────────────────────────────────────────────────────────

EmbeddingVector = list[float] | np.ndarray
EmbeddingBatch = list[EmbeddingVector]
EmbeddingOutput = EmbeddingBatch | np.ndarray  # list of vectors or single 2D array
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


GenerateEmbeddingsReturnType = EmbeddingOutput  # kept for backward compatibility
