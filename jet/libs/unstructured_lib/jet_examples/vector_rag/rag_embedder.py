"""LlamaCppEmbedder: generic wrapper for llama.cpp embedding server (OpenAI compatible)."""

import os

from jet.adapters.llama_cpp.parallel_embeddings import embed_batch, embed_single
from jet.adapters.llama_cpp.types import LLAMACPP_EMBED_KEYS

DEFAULT_EMBED_URL = os.getenv("LLAMA_CPP_EMBED_URL")
DEFAULT_MODEL_NAME: LLAMACPP_EMBED_KEYS = os.getenv("LLAMA_CPP_EMBED_MODEL")
DEFAULT_BATCH_SIZE = 32  # sensible default


class LlamaCppEmbedder:
    """Reusable embedder - works with any OpenAI-compatible server (llama.cpp embed URL)."""

    def __init__(
        self,
        embed_model: LLAMACPP_EMBED_KEYS = DEFAULT_MODEL_NAME,
        embed_url: str = DEFAULT_EMBED_URL,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        self.embed_model = embed_model
        self.embed_url = embed_url
        self.batch_size = batch_size
        if not embed_url:
            raise ValueError("LLAMA_CPP_EMBED_URL env var required (or pass embed_url)")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Batch embed - generic, handles empty safely."""
        return embed_batch(
            texts,
            self.embed_model,
            return_format="list",
            batch_size=self.batch_size,
        )

    def embed_query(self, text: str) -> list[float]:
        """Single query embed (reusable in retriever)."""
        return embed_single(text, self.embed_model, return_format="list")
