"""
Centralized configuration for the Repo Agent system.
Adjust these values to match your local server ports and preferences.
"""

from dataclasses import dataclass
from pathlib import Path

from jet.adapters.llama_cpp.config import (
    EMBED_BASE_URL,
    EMBED_MODEL_LG,
    LLM_BASE_URL,
    LLM_MODEL,
    RERANK_BASE_URL,
    RERANK_MODEL,
)


@dataclass
class ServerConfig:
    # Qwen 2B Server (OpenAI-compatible endpoint)
    LLM_URL: str = LLM_BASE_URL

    # Nomic Embed v2 Server
    EMBED_URL: str = EMBED_BASE_URL

    # BGE Reranker v2 Server
    RERANK_URL: str = RERANK_BASE_URL


@dataclass
class ModelConfig:
    LLM_MODEL: str = LLM_MODEL
    EMBED_MODEL: str = EMBED_MODEL_LG
    RERANK_MODEL: str = RERANK_MODEL


@dataclass
class RetrievalConfig:
    TOP_K_INITIAL: int = 50  # Candidates from vector search
    TOP_K_FINAL: int = 5  # Final results after reranking
    DISTANCE_THRESHOLD: float = 0.7  # Max cosine distance (0-1)
    USE_HYBRID: bool = True  # Enable keyword boosting if supported


@dataclass
class ChunkingConfig:
    CHUNK_SIZE: int = 800  # Tokens per chunk
    CHUNK_OVERLAP: int = 100  # Overlap tokens
    SUPPORTED_EXTS: list = None

    def __post_init__(self):
        if self.SUPPORTED_EXTS is None:
            self.SUPPORTED_EXTS = [
                ".py",
                ".js",
                ".ts",
                ".jsx",
                ".tsx",
                ".java",
                ".cpp",
                ".c",
                ".h",
                ".hpp",
                ".go",
                ".rs",
                ".rb",
                ".php",
                ".swift",
                ".md",
                ".txt",
                ".json",
                ".yaml",
                ".yml",
                ".toml",
            ]


@dataclass
class DBConfig:
    PATH: str = (
        Path("~/.cache/projects/repo-agent/repo_index.lancedb")
        .expanduser()
        .resolve()
        .absolute()
    )
    TABLE_NAME: str = "code_chunks"
    DIMENSIONS: int = 768  # Nomic v2 dimensions


# Global Instance
servers = ServerConfig()
models = ModelConfig()
retrieval = RetrievalConfig()
chunking = ChunkingConfig()
db = DBConfig()
