"""
Centralized configuration for the Repo Agent system.
Adjust these values to match your local server ports and preferences.
"""

from dataclasses import dataclass, field
from pathlib import Path

from jet.adapters.llama_cpp.config import (
    EMBED_BASE_URL,
    EMBED_MODEL_LG,
    LLM_BASE_URL,
    LLM_MODEL,
    PHOENIX_BASE_URL,
    RERANK_BASE_URL,
    RERANK_MODEL,
)


@dataclass
class ServerConfig:
    LLM_URL: str = LLM_BASE_URL
    EMBED_URL: str = EMBED_BASE_URL
    RERANK_URL: str = RERANK_BASE_URL
    PHOENIX_URL: str = PHOENIX_BASE_URL


@dataclass
class ModelConfig:
    LLM_MODEL: str = LLM_MODEL
    EMBED_MODEL: str = EMBED_MODEL_LG
    RERANK_MODEL: str = RERANK_MODEL


@dataclass
class RetrievalConfig:
    TOP_K_INITIAL: int = 50
    TOP_K_FINAL: int = 5
    DISTANCE_THRESHOLD: float = 0.7
    USE_HYBRID: bool = True


@dataclass
class ChunkingConfig:
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100
    SUPPORTED_EXTS: list = field(
        default_factory=lambda: [
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
            ".mdx",
            ".txt",
            ".json",
            ".yaml",
            ".yml",
            ".toml",
        ]
    )


@dataclass
class DBConfig:
    PATH: str = (
        Path("~/.cache/projects/repo-agent/repo_index.lancedb")
        .expanduser()
        .resolve()
        .absolute()
    )
    TABLE_NAME: str = "code_chunks"
    DIMENSIONS: int = 768  # Will be auto-detected if possible


servers = ServerConfig()
models = ModelConfig()
retrieval = RetrievalConfig()
chunking = ChunkingConfig()
db = DBConfig()
