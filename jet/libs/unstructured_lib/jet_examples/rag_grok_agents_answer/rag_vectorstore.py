"""ChromaVectorStore: generic local vector store using pre-computed embeddings."""

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

    def add_documents(
        self, documents: ChunkList, embeddings: list[list[float]]
    ) -> None:
        """Add with pre-computed embeds - DRY, handles empty."""
        if not documents:
            return
        # temporary debug
        console.print(
            f"[yellow]DEBUG ChromaVectorStore.add_documents: {len(documents)} chunks[/yellow]"
        )
        ids = [str(uuid4()) for _ in documents]
        texts = [d["text"] for d in documents]
        # Normalize empty dicts – fixes ChromaDB "0 metadata attributes" error
        metas = [d.get("metadata") or None for d in documents]
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
