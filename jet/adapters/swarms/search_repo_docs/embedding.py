"""Custom LlamaIndex BaseEmbedding backed by jet.adapters.llama_cpp.embed_utils."""

from __future__ import annotations

from typing import Any, List

from jet.adapters.llama_cpp.config import (
    EMBED_DOC_PREFIX,
    EMBED_MODEL,
    EMBED_QUERY_PREFIX,
)
from jet.adapters.llama_cpp.embed_utils import embed_batch, embed_single
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.embeddings import BaseEmbedding


class LlamaCppEmbedding(BaseEmbedding):
    """
    LlamaIndex-compatible embedding model backed by jet.adapters.llama_cpp.embed_utils.

    Benefits over OpenAIEmbedding:
    - Automatic deduplication of identical chunks before embedding
    - Parallel batched embedding with ThreadPoolExecutor
    - Pre-flight server health check
    - Rich progress bars with RTT reporting
    """

    model_name: str = Field(
        default=EMBED_MODEL, description="Embedding model identifier"
    )
    query_prefix: str = Field(
        default=EMBED_QUERY_PREFIX, description="Prefix for query texts"
    )
    text_prefix: str = Field(
        default=EMBED_DOC_PREFIX, description="Prefix for document texts"
    )
    batch_size: int = Field(
        default=64, description="Texts per batch for RTT amortization"
    )
    max_workers: int = Field(default=6, description="Concurrent embedding threads")
    show_progress: bool = Field(default=True, description="Show Rich progress bar")

    _model: str = PrivateAttr()

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._model = self.model_name

    @classmethod
    def class_name(cls) -> str:
        return "LlamaCppEmbedding"

    def _get_query_embedding(self, query: str) -> List[float]:
        """Embed a single query string with query prefix."""
        prefixed = f"{self.query_prefix}{query}" if self.query_prefix else query
        return embed_single(text=prefixed, model=self._model, return_format="list")

    def _get_text_embedding(self, text: str) -> List[float]:
        """Embed a single document text with document prefix."""
        prefixed = f"{self.text_prefix}{text}" if self.text_prefix else text
        return embed_single(text=prefixed, model=self._model, return_format="list")

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple document texts with deduplication and parallel batching.

        This is where embed_utils shines: identical chunks are embedded once
        and mapped back to all original positions, avoiding redundant API calls.
        """
        if not texts:
            return []
        prefixed = [f"{self.text_prefix}{t}" if self.text_prefix else t for t in texts]
        result = embed_batch(
            texts=prefixed,
            model=self._model,
            max_workers=self.max_workers,
            show_progress=self.show_progress,
            return_format="list",
            batch_size=self.batch_size,
            progress_description="Embedding chunks",
        )
        # embed_batch returns list[list[float]] when return_format="list"
        return result

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Async fallback — delegates to sync (embed_utils is thread-based, not async)."""
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)
