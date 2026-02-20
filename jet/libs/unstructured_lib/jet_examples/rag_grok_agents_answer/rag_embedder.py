"""LlamaCppEmbedder: generic wrapper for llama.cpp embedding server (OpenAI compatible)."""

import os

from openai import OpenAI


class LlamaCppEmbedder:
    """Reusable embedder - works with any OpenAI-compatible server (llama.cpp embed URL)."""

    def __init__(self, embed_url: str | None = None):
        url = embed_url or os.getenv("LLAMA_CPP_EMBED_URL")
        if not url:
            raise ValueError("LLAMA_CPP_EMBED_URL env var required (or pass embed_url)")
        self.client = OpenAI(
            base_url=url.rstrip("/") + "/v1" if not url.endswith("/v1") else url,
            api_key="sk-no-key-required",
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Batch embed - generic, handles empty safely."""
        if not texts:
            return []
        response = self.client.embeddings.create(
            input=texts, model="dummy"
        )  # model ignored by llama.cpp
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        """Single query embed (reusable in retriever)."""
        return self.embed_documents([text])[0]
