# jet_python_modules/jet/adapters/langchain/embed_llama_cpp.py
from __future__ import annotations

from typing import Any, List, Literal, Optional

from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, Field

from jet.adapters.llama_cpp.embeddings import LlamacppEmbedding


class LlamaCppEmbeddings(BaseModel, Embeddings):
    """
    LangChain embeddings wrapper for LlamacppEmbedding.
    Supports both list and numpy return formats with batch processing and caching.
    """
    model_config = {"arbitrary_types_allowed": True}  # Allow LlamacppEmbedding

    model: str = Field(default="embeddinggemma", description="Embedding model name")
    base_url: str = Field(
        default="http://shawn-pc.local:8081/v1",
        description="Base URL of the llama.cpp embedding server",
    )
    max_retries: int = Field(default=3, ge=0, description="Max retries on HTTP failure")
    cache_backend: Literal["memory", "file", "sqlite"] = Field(
        default="sqlite", description="Cache storage backend"
    )
    cache_ttl: Optional[int] = Field(default=None, ge=1, description="Cache TTL in seconds")
    cache_max_size: int = Field(default=10000, ge=1, description="Maximum cache size")
    use_cache: bool = Field(default=False, description="Enable embedding caching")
    use_dynamic_batch_sizing: bool = Field(
        default=False, description="Dynamically adjust batch size"
    )
    batch_size: int = Field(default=32, ge=1, description="Batch size for embedding calls")
    return_format: Literal["list", "numpy"] = Field(
        default="list", description="Output format: list of lists or numpy array"
    )
    show_progress: bool = Field(default=True, description="Show progress bar")
    embedder: LlamacppEmbedding = Field(default=None, exclude=True, repr=False)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.embedder = LlamacppEmbedding(
            model=self.model,
            base_url=self.base_url,
            max_retries=self.max_retries,
            cache_backend=self.cache_backend,
            cache_ttl=self.cache_ttl,
            cache_max_size=self.cache_max_size,
            use_cache=self.use_cache,
            use_dynamic_batch_sizing=self.use_dynamic_batch_sizing,
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        result = self.embedder(
            inputs=texts,
            return_format=self.return_format,
            batch_size=self.batch_size,
            show_progress=self.show_progress,
            use_cache=self.use_cache,
            use_dynamic_batch_sizing=self.use_dynamic_batch_sizing,
        )
        return result if self.return_format == "list" else result.tolist()

    def embed_query(self, text: str) -> List[float]:
        result = self.embedder(
            inputs=text,
            return_format=self.return_format,
            batch_size=1,
            show_progress=False,
            use_cache=self.use_cache,
        )
        return result[0] if self.return_format == "list" else result[0].tolist()

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        # llama.cpp embedding server does not support async; fall back to sync
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> List[float]:
        # llama.cpp embedding server does not support async; fall back to sync
        return self.embed_query(text)