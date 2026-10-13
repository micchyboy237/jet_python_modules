"""LangChain Embeddings adapter for llama.cpp using jet embed utilities."""

from typing import List, Optional

import numpy as np
from jet.adapters.llama_cpp.config import EMBED_MODEL
from jet.adapters.llama_cpp.embed_utils import embed as jet_embed
from langchain_core.embeddings import Embeddings
from pydantic import Field


class EmbedLlamaCpp(Embeddings):
    """Embeddings wrapper for llama.cpp using jet's optimized embedding utilities.

    This class integrates llama.cpp embedding models into LangChain by leveraging
    jet's built-in batching, deduplication, and progress tracking features.
    """

    model_name: str = Field(default=EMBED_MODEL, alias="model")
    """The name of the embedding model to use."""

    prefix: Optional[str] = Field(default=None)
    """Optional prefix to add to each text before embedding (e.g., for query/doc distinction)."""

    batch_size: int = 64
    """Batch size for embedding documents."""

    max_workers: int = 6
    """Number of workers for parallel embedding."""

    show_progress: bool = False
    """Whether to show a progress bar during embedding."""

    class Config:
        allow_population_by_field_name = True

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Internal helper to call jet embed utility."""
        if not texts:
            return []

        # Use jet's embed function which handles batching, deduplication, and API calls
        result = jet_embed(
            text=texts,
            model=self.model_name,
            return_format="list",
            max_workers=self.max_workers,
            show_progress=self.show_progress,
            batch_size=self.batch_size,
            prefix=self.prefix,
        )

        # jet_embed returns list[list[float]] or np.ndarray
        if isinstance(result, np.ndarray):
            return result.tolist()
        return result

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        return self._embed_texts(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        # For single query, we can still use the batch helper or call directly
        # jet_embed handles str input via embed_single internally
        result = jet_embed(
            text=text,
            model=self.model_name,
            return_format="list",
            prefix=self.prefix,
        )

        if isinstance(result, np.ndarray):
            return result.tolist()
        return result

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Asynchronous Embed search docs.

        Note: jet's embed_utils is currently synchronous/thread-pool based.
        We run it in an executor to maintain async compatibility without blocking.
        """
        from langchain_core.runnables.config import run_in_executor

        return await run_in_executor(None, self.embed_documents, texts)

    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronous Embed query text."""
        from langchain_core.runnables.config import run_in_executor

        return await run_in_executor(None, self.embed_query, text)
