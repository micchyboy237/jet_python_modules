"""ChromaVectorStore: generic local vector store using pre-computed embeddings."""

from typing import Any
from uuid import uuid4

import chromadb
from rich.console import Console

from .rag_document import ChunkList

console = Console()


class ChromaVectorStore:
    """Reusable Chroma wrapper - persistent by default, metadata-aware."""

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "default_rag",
    ):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def _sanitize_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Private helper: convert lists/dicts to primitives so ChromaDB accepts them (generic, DRY)."""
        if not metadata:
            return {}
        sanitized: dict[str, Any] = {}
        for k, v in metadata.items():
            if isinstance(v, (list, tuple)):
                sanitized[k] = ", ".join(str(item) for item in v)
            elif isinstance(v, dict):
                sanitized[k] = str(v)
            else:
                sanitized[k] = v
        return sanitized

    def add_documents(
        self, documents: ChunkList, embeddings: list[list[float]]
    ) -> None:
        """Add with pre-computed embeds - DRY, handles empty."""
        if not documents:
            return
        # temporary debug (remove after you confirm working)
        console.print(
            f"[yellow]DEBUG ChromaVectorStore.add_documents: sanitizing {len(documents)} chunks[/yellow]"
        )
        ids = [str(uuid4()) for _ in documents]
        texts = [d["text"] for d in documents]
        metas = [self._sanitize_metadata(d.get("metadata") or {}) for d in documents]
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metas,
        )

    def similarity_search(self, query_embedding: list[float], k: int = 5) -> ChunkList:
        """Retrieve top-k - reconstructs ChunkList."""
        if not query_embedding:
            return []
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas"],
        )
        chunks: ChunkList = []
        docs = results.get("documents") or [[]]
        metas = results.get("metadatas") or [[]]
        for doc, meta in zip(docs[0], metas[0]):
            chunks.append({"text": doc, "metadata": meta or {}})
        return chunks
