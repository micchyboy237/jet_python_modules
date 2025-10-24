from typing import List, Optional, Literal
from langchain_ollama import OllamaEmbeddings as BaseOllamaEmbeddings
from jet.logger import logger
from jet.llm.utils.embeddings import get_embedding_function
from pydantic import Field

class OllamaEmbeddings(BaseOllamaEmbeddings):
    batch_size: int = Field(default=32, description="Batch size for embedding processing")
    return_format: Literal["list", "numpy"] = Field(default="list", description="Format of the returned embeddings")
    use_cache: bool = Field(default=False, description="Whether to cache embeddings for reuse")

    def __init__(
        self,
        base_url: Optional[str] = "http://jethros-macbook-air.local:11434",
        batch_size: int = 32,
        return_format: Literal["list", "numpy"] = "list",
        use_cache: bool = False,
        **kwargs
    ):
        """Initialize with default base_url, batch_size, return_format, and use_cache."""
        super().__init__(base_url=base_url, **kwargs)
        self.batch_size = batch_size
        self.return_format = return_format
        self.use_cache = use_cache

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs using utils.embeddings with progress tracking."""
        if not self._client:
            msg = (
                "Ollama client is not initialized. "
                "Please ensure Ollama is running and the model is loaded."
            )
            raise ValueError(msg)

        embed_func = get_embedding_function(
            model_name=self.model,
            batch_size=self.batch_size,
            return_format=self.return_format,
            url=self.base_url,
            use_cache=self.use_cache,
        )

        try:
            embeddings = embed_func(texts)
            return embeddings
        except Exception as e:
            logger.error(f"Error embedding documents: {str(e)}")
            raise

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async embed search docs using utils.embeddings with progress tracking."""
        if not self._async_client:
            msg = (
                "Ollama async client is not initialized. "
                "Please ensure Ollama is running and the model is loaded."
            )
            raise ValueError(msg)

        embed_func = get_embedding_function(
            model_name=self.model,
            batch_size=self.batch_size,
            return_format=self.return_format,
            url=self.base_url,
            use_cache=self.use_cache,  # Added to ensure async method uses cache
        )

        try:
            embeddings = embed_func(texts)
            return embeddings
        except Exception as e:
            logger.error(f"Error embedding documents asynchronously: {str(e)}")
            raise
