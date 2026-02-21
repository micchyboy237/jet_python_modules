"""ChromaVectorStore: generic local vector store using pre-computed embeddings + BM25 + hybrid support."""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import chromadb
from rank_bm25 import BM25Okapi
from rich.console import Console

from .rag_document import Chunk, ChunkList

console = Console()


def _simple_tokenize(text: str) -> List[str]:
    """Minimal tokenizer suitable for basic BM25 — can be improved later (stemming, stop words, etc.)."""
    return text.lower().split()


class ChromaVectorStore:
    """Persistent Chroma vector store with optional BM25 keyword search and hybrid RRF fusion."""

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "default_rag",
    ):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name=collection_name)

        # BM25 support structures (in-memory)
        self.chunks: Dict[str, Chunk] = {}  # id → full chunk dict
        self.tokenized_docs: List[List[str]] = []  # parallel list for BM25
        self.id_order: List[str] = []  # id parallel to tokenized_docs
        self.bm25: Optional[BM25Okapi] = None

    def _sanitize_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Convert non-primitive values to strings so Chroma accepts them."""
        if not metadata:
            return {}
        sanitized: Dict[str, Any] = {}
        for k, v in metadata.items():
            if isinstance(v, (list, tuple)):
                sanitized[k] = ", ".join(str(item) for item in v)
            elif isinstance(v, dict):
                sanitized[k] = str(v)
            else:
                sanitized[k] = v
        return sanitized

    def add_documents(
        self,
        documents: ChunkList,
        embeddings: List[List[float]],
    ) -> None:
        """Add documents with pre-computed embeddings to Chroma + build BM25 index."""
        if not documents:
            console.print("[yellow]No documents to add[/yellow]")
            return

        if len(documents) != len(embeddings):
            raise ValueError("Number of documents and embeddings must match")

        console.print(
            f"[yellow]Adding {len(documents)} documents to Chroma + BM25[/yellow]"
        )

        ids = [str(uuid4()) for _ in documents]
        texts = [d["text"] for d in documents]
        metadatas = [self._sanitize_metadata(d.get("metadata", {})) for d in documents]

        # Add to Chroma
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        # Build BM25 structures
        for chunk_id, doc in zip(ids, documents):
            self.chunks[chunk_id] = doc
            tokens = _simple_tokenize(doc["text"])
            self.tokenized_docs.append(tokens)
            self.id_order.append(chunk_id)

        if self.tokenized_docs:
            self.bm25 = BM25Okapi(self.tokenized_docs)
            console.print(
                f"[green]BM25 index built with {len(self.tokenized_docs)} documents[/green]"
            )

    def vector_search(
        self,
        query_embedding: List[float],
        k: int = 5,
    ) -> ChunkList:
        """Perform dense vector similarity search (original behavior)."""
        if not query_embedding:
            return []

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas"],
        )

        chunks: ChunkList = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        ids_list = results.get("ids", [[]])[0]

        for doc_text, meta, doc_id in zip(docs, metas, ids_list):
            chunk: Chunk = {"text": doc_text, "metadata": meta or {}}
            # Optional: chunk["id"] = doc_id   # if you want to expose id downstream
            chunks.append(chunk)

        return chunks

    def keyword_search(
        self,
        query_text: str,
        k: int = 5,
    ) -> ChunkList:
        """Pure BM25 keyword-based retrieval."""
        if not self.bm25 or not self.tokenized_docs:
            console.print(
                "[yellow]BM25 index not available — no documents ingested yet[/yellow]"
            )
            return []

        query_tokens = _simple_tokenize(query_text)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        # Get indices sorted by descending score
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
            :k
        ]

        return [self.chunks[self.id_order[idx]] for idx in top_indices]

    def hybrid_rrf_search(
        self,
        query_embedding: List[float],
        query_text: str,
        k: int = 5,
        oversample_factor: int = 3,
        rrf_constant: int = 60,
    ) -> ChunkList:
        """
        Hybrid retrieval: run both vector and BM25 → fuse rankings with Reciprocal Rank Fusion.
        Returns top-k fused results.
        """
        if not self.bm25 or not self.tokenized_docs:
            console.print(
                "[yellow]No BM25 index — falling back to pure vector search[/yellow]"
            )
            return self.vector_search(query_embedding, k)

        candidates_k = max(5, k * oversample_factor)

        # 1. Vector retrieval
        vec_chunks = self.vector_search(query_embedding, candidates_k)

        # 2. BM25 retrieval
        bm25_chunks = self.keyword_search(query_text, candidates_k)

        # 3. Build ranked lists of document ids (we use text as proxy key — assumes unique chunks)
        #    In production consider using deterministic content hash or exposing real ids
        vec_ranked: List[Tuple[str, int]] = []
        for rank, chunk in enumerate(vec_chunks, start=1):
            vec_ranked.append((chunk["text"], rank))

        bm25_ranked: List[Tuple[str, int]] = []
        for rank, chunk in enumerate(bm25_chunks, start=1):
            bm25_ranked.append((chunk["text"], rank))

        # 4. Reciprocal Rank Fusion
        fusion_scores: Dict[str, float] = defaultdict(float)

        for doc_key, rank in vec_ranked:
            fusion_scores[doc_key] += 1.0 / (rrf_constant + rank)

        for doc_key, rank in bm25_ranked:
            fusion_scores[doc_key] += 1.0 / (rrf_constant + rank)

        # Sort by fused score descending
        sorted_keys = sorted(
            fusion_scores.keys(), key=lambda key: fusion_scores[key], reverse=True
        )[:k]

        # Reconstruct chunks (lookup by text — simple but works if chunks are unique)
        retrieved: ChunkList = []
        text_to_chunk: Dict[str, Chunk] = {c["text"]: c for c in self.chunks.values()}

        for key in sorted_keys:
            if key in text_to_chunk:
                retrieved.append(text_to_chunk[key])

        return retrieved
