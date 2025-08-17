from typing import List, Dict, Any, Optional
from autogen_core.memory import Memory, MemoryContent, MemoryMimeType, MemoryQueryResult, UpdateContextResult
from autogen_core.memory._base_memory import ChatCompletionContext, CancellationToken
from autogen_core.models import LLMMessage, UserMessage, AssistantMessage
from autogen_ext.memory.mem0 import Mem0Memory
from sentence_transformers import SentenceTransformer
from jet.models.model_registry.transformers.mlx_model_registry import MLXModelRegistry
from jet.models.model_registry.transformers.sentence_transformer_registry import SentenceTransformerRegistry
from jet.models.model_types import EmbedModelType, LLMModelType
from jet.models.utils import get_context_size, get_embedding_size, resolve_model_value
from mem0 import Memory as Memory0
import numpy as np
import os


class MemoryManager(Memory):
    """Manages memory operations using Mem0Memory with MLX LLM and Sentence Transformers."""

    def __init__(self, user_id: Optional[str] = None, limit: int = 5,
                 llm_model_path: LLMModelType = "qwen3-1.7b-4bit",
                 embedder_model_path: EmbedModelType = "all-MiniLM-L6-v2",
                 **kwargs):
        """Initialize the Mem0Memory client with local MLX and Sentence Transformers.
        Args:
            user_id: Unique identifier for the user. If None, a UUID is generated.
            limit: Maximum number of memories to retrieve in queries.
            llm_model_path: Path to the local MLX model directory.
            embedder_model_path: Path to the local Sentence Transformer model directory.
        """
        super().__init__()
        self.embedder = SentenceTransformerRegistry.load_model(
            embedder_model_path, device="mps")
        self.config = {
            "llm": {
                "provider": "mlx",
                "config": {
                    "model": resolve_model_value(llm_model_path),
                    **kwargs
                },
            },
            "embedder": {
                "provider": "huggingface",
                "config": {
                    "model": resolve_model_value(embedder_model_path),
                    "embedding_dims": get_embedding_size(embedder_model_path),
                },
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": "memories",
                    "host": "jethros-macbook-air.local",
                    "port": 6333,
                    "embedding_model_dims": get_embedding_size(embedder_model_path),
                },
            },
            "version": "v1.1",
        }
        self.memory = Mem0Memory(
            user_id=user_id,
            limit=limit,
            is_cloud=False,
            config=self.config,
        )

    async def add(self, content: MemoryContent, cancellation_token: CancellationToken | None = None) -> None:
        """Add a memory to the store.
        Args:
            content: The memory content to add.
            cancellation_token: Optional token to cancel operation.
        """
        await self.memory.add(content, cancellation_token=cancellation_token)

    async def query(self, query: str | MemoryContent, cancellation_token: CancellationToken | None = None, **kwargs: Any) -> MemoryQueryResult:
        """Search for memories based on a query.
        Args:
            query: The search query string or MemoryContent.
            cancellation_token: Optional token to cancel operation.
            **kwargs: Additional implementation-specific parameters.
        Returns:
            MemoryQueryResult containing memory entries with relevance scores.
        """
        result = await self.memory.query(query, cancellation_token=cancellation_token, **kwargs)
        return MemoryQueryResult(results=result.results)

    async def update_context(self, model_context: ChatCompletionContext) -> UpdateContextResult:
        """Update the provided model context using relevant memory content.
        Args:
            model_context: The context to update.
        Returns:
            UpdateContextResult containing relevant memories.
        """
        messages = model_context.get_messages()
        query = ""
        if messages:
            last_message = messages[-1]
            if isinstance(last_message, (UserMessage, AssistantMessage)):
                query = last_message.content if isinstance(
                    last_message.content, str) else ""
        result = await self.query(query)
        return UpdateContextResult(memories=result)

    async def clear(self) -> None:
        """Clear all entries from memory."""
        if hasattr(self.memory, 'clear'):
            await self.memory.clear()
        else:
            raise NotImplementedError(
                "Mem0Memory does not support clear operation")

    async def close(self) -> None:
        """Clean up any resources used by the memory implementation."""
        if hasattr(self.memory, 'close'):
            await self.memory.close()
        else:
            # No-op if close is not supported
            pass
