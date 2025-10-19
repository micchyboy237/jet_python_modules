"""
RAG Retriever integration with Stanza-based preprocessing.
Uses sentence-transformers for dense embeddings and FAISS for fast retrieval.
"""

from __future__ import annotations
import faiss
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
from sentence_transformers import SentenceTransformer


@dataclass
class ContextChunk:
    """Container for preprocessed RAG context."""
    text: str
    salience: float
    entities: List[str]
    sentence_indices: List[int]


class RagRetrieverFAISS:
    """
    A simple retriever built on FAISS for Stanza RAG chunks.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunk_store: List[ContextChunk] = []

    # ----------------------------------------------------------------------
    def build_index(self, chunks: List[ContextChunk]) -> None:
        """Embed and index the given chunks."""
        if not chunks:
            raise ValueError("No chunks provided to build index.")

        self.chunk_store = chunks
        embeddings = self.model.encode([c.text for c in chunks], convert_to_numpy=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

    # ----------------------------------------------------------------------
    def retrieve(self, query: str, top_k: int = 2) -> List[Tuple[float, ContextChunk]]:
        """Retrieve top_k most similar chunks for the query."""
        if self.index is None:
            raise RuntimeError("FAISS index not built. Call build_index first.")

        query_emb = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_emb, top_k)
        results = []
        for i, dist in zip(indices[0], distances[0]):
            chunk = self.chunk_store[int(i)]
            results.append((float(dist), chunk))
        return results

    # ----------------------------------------------------------------------
    def describe_result(self, query: str, top_k: int = 2) -> Dict[str, Any]:
        """Return human-readable retrieval summary."""
        results = self.retrieve(query, top_k)
        return {
            "query": query,
            "results": [
                {
                    "rank": i + 1,
                    "distance": round(dist, 4),
                    "salience": round(chunk.salience, 2),
                    "preview": chunk.text[:120] + "...",
                    "entities": chunk.entities,
                }
                for i, (dist, chunk) in enumerate(results)
            ],
        }
