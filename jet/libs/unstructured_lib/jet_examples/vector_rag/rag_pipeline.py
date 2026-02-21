"""RAGPipeline: top-level orchestrator - composable, small methods."""

import os
from typing import Literal

from rich.console import Console

from .rag_document import ChunkList
from .rag_embedder import LlamaCppEmbedder
from .rag_llm import LlamaCppLLM
from .rag_processor import DocumentProcessor
from .rag_vectorstore import ChromaVectorStore

console = Console()


class RAGPipeline:
    """Generic pipeline - dependency injection for testability/reuse."""

    def __init__(
        self,
        processor: DocumentProcessor | None = None,
        embedder: LlamaCppEmbedder | None = None,
        llm: LlamaCppLLM | None = None,
        vector_store: ChromaVectorStore | None = None,
    ):
        self.processor = processor or DocumentProcessor()
        self.embedder = embedder or LlamaCppEmbedder()
        self.llm = llm or LlamaCppLLM()
        self.vector_store = vector_store or ChromaVectorStore()

    def ingest(self, file_paths_or_dir: str | list[str]) -> None:
        """Ingest files or dir - uses tqdm + rich for visibility."""
        if isinstance(file_paths_or_dir, str) and os.path.isdir(file_paths_or_dir):
            chunks: ChunkList = self.processor.process_directory(file_paths_or_dir)
        else:
            paths = (
                [file_paths_or_dir]
                if isinstance(file_paths_or_dir, str)
                else file_paths_or_dir
            )
            chunks = []
            for p in paths:
                chunks.extend(self.processor.process_file(p))
        if not chunks:
            console.print("[yellow]No chunks to ingest[/yellow]")
            return
        embeddings = self.embedder.embed_documents([c["text"] for c in chunks])
        self.vector_store.add_documents(chunks, embeddings)
        console.print(f"[green]Ingested {len(chunks)} chunks successfully[/green]")

    def query(
        self,
        question: str,
        k: int = 5,
        temperature: float = 0.0,
        mode: Literal["vector", "bm25", "hybrid-rrf"] = "vector",
    ) -> str:
        """Full RAG query - embed -> retrieve -> generate. Generic prompt (overrideable via subclass)."""
        query_emb = self.embedder.embed_query(question)
        if mode == "vector":
            retrieved = self.vector_store.vector_search(query_emb, k=k)
        elif mode == "bm25":
            retrieved = self.vector_store.keyword_search(question, k=k)
        elif mode == "hybrid-rrf":
            retrieved = self.vector_store.hybrid_rrf_search(query_emb, question, k=k)
        else:
            raise ValueError(f"Unknown retrieval mode: {mode}")

        if not retrieved:
            return "No relevant documents found."
        context = "\n\n---\n\n".join([c["text"] for c in retrieved])
        system_prompt = (
            "You are a helpful, accurate assistant. Answer only using the provided context. "
            "If unsure, say 'I don't have enough information'."
        )
        user_prompt = f"Context:\n{context}\n\nQuestion: {question}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        answer = self.llm.generate(messages, temperature=temperature)
        console.print(f"[blue]Retrieved {len(retrieved)} chunks[/blue]")
        return answer
