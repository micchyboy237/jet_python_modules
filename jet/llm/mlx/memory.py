from typing import List, Dict, Any, Optional
from autogen_ext.memory.mem0 import Mem0Memory
from autogen_core.memory import MemoryContent, MemoryMimeType
from sentence_transformers import SentenceTransformer
from jet.models.model_registry.transformers.mlx_model_registry import MLXModelRegistry
from jet.models.model_registry.transformers.sentence_transformer_registry import SentenceTransformerRegistry
from jet.models.model_types import EmbedModelType, LLMModelType
from jet.models.utils import get_context_size, get_embedding_size, resolve_model_value
from mem0 import Memory as Memory0
import numpy as np
import os


class MemoryManager:
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
        # # Load MLX model
        # self.llm_model = MLXModelRegistry.load_model(llm_model_path, **kwargs)
        # self.llm_tokenizer = self.llm_model.tokenizer

        # Load Sentence Transformer model
        self.embedder = SentenceTransformerRegistry.load_model(
            embedder_model_path, device="mps")

        # Configure Mem0Memory with a supported provider (ollama as placeholder)
        self.config = {
            # "llm": {
            #     "provider": "ollama",  # Placeholder to pass validation
            #     "config": {
            #         "model": "llama3.1",  # Dummy model name
            #         "ollama_base_url": "http://localhost:11434",  # Dummy URL
            #     },
            # },
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

    #     # Override Memory0's LLM and embedder methods
    #     self._patch_memory_client()

    # def _patch_memory_client(self):
    #     """Patch the Memory0 client to use custom MLX LLM and Sentence Transformer embedder."""

    #     def custom_generate(self, prompt: str, **kwargs) -> str:
    #         response = self.llm_model.generate(
    #             prompt=prompt,
    #             max_tokens=512,
    #             temperature=0.7,
    #             verbose=False,
    #         )
    #         return response["content"]

    #     def custom_embed(self, texts: List[str], **kwargs) -> List[np.ndarray]:
    #         embeddings = self.embedder.encode(
    #             texts, normalize_embeddings=True, convert_to_numpy=True)
    #         return [np.array(emb) for emb in embeddings]

    #     # Bind custom methods to the Memory0 instance
    #     self.memory._client.generate = custom_generate.__get__(
    #         self, MemoryManager)
    #     self.memory._client.embed = custom_embed.__get__(self, MemoryManager)

    async def add(self, content: str, metadata: Optional[Dict[str, Any]] = None, mime_type: MemoryMimeType | str = "text/plain") -> None:
        """Add a memory to the store.

        Args:
            content: The content to store as a memory.
            metadata: Optional metadata to associate with the memory.
        """
        memory_content = MemoryContent(
            content=content, mime_type=mime_type, metadata=metadata or {})
        await self.memory.add(memory_content)

    async def search(self, query: str) -> List[Dict[str, Any]]:
        """Search for memories based on a query.

        Args:
            query: The search query string.

        Returns:
            List of memory dictionaries with content and metadata.
        """
        result = await self.memory.query(query)
        return [
            {"content": mem.content, "metadata": mem.metadata}
            for mem in result.results
        ]
