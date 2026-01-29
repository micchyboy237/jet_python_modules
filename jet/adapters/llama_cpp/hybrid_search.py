from __future__ import annotations
import math
from dataclasses import dataclass
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from numpy.typing import NDArray
from rank_bm25 import BM25Okapi

from jet.adapters.llama_cpp.types import LLAMACPP_EMBED_KEYS
from jet.adapters.llama_cpp.embeddings import (
    LlamacppEmbedding,
    EmbeddingVector,
    cosine_similarity,  # ← reused
    SearchResultType,  # can be reused if needed later
)


@dataclass(frozen=True)
class SearchResult:
    """Simple container for a retrieved item + score"""

    item: Any
    score: float

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, SearchResult):
            return NotImplemented
        return self.score < other.score


def normalize_embedding(vec: EmbeddingVector) -> NDArray[np.float32]:
    """Normalize vector to unit length (safe against zero vectors)."""
    arr = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm < 1e-12:
        return arr  # avoid division by zero — return as-is
    return arr / norm


class VectorRetriever:
    """Simple in-memory vector store + dense retriever — aligned with embeddings.py"""

    def __init__(self, embedder: LlamacppEmbedding):
        self.embedder = embedder
        self.documents: List[Any] = []
        self.embeddings: NDArray[np.float32] = np.empty((0, 0), dtype=np.float32)

    def index(self, documents: List[Any], texts: List[str]) -> None:
        """Index documents with their corresponding texts"""
        if len(documents) != len(texts):
            raise ValueError("documents and texts must have same length")
        if not texts:
            return

        # Reuse embedding method from embeddings.py
        # We ask for list format → easier to stack safely
        raw_embeddings = self.embedder.embed(
            texts,
            return_format="list",  # consistent with embeddings.py style
            batch_size=32,  # can be made configurable later
            show_progress=False,  # usually silent during indexing
        )

        # Convert to normalized numpy array
        normalized = [normalize_embedding(emb) for emb in raw_embeddings]
        new_emb_array = np.array(normalized, dtype=np.float32)

        # Append documents & embeddings
        self.documents.extend(documents)

        if self.embeddings.size == 0:
            self.embeddings = new_emb_array
        else:
            self.embeddings = np.vstack([self.embeddings, new_emb_array])

    def search(self, query: str, k: int = 50) -> List[SearchResult]:
        if self.embeddings.size == 0 or not self.documents:
            return []

        # Single query → get list of one embedding
        [raw_q_emb] = self.embedder.embed(
            [query],
            return_format="list",
            batch_size=1,
            show_progress=False,
        )

        q_emb = normalize_embedding(raw_q_emb)

        # Vectorized cosine similarity (assuming embeddings are already normalized)
        # scores = self.embeddings @ q_emb   # faster when pre-normalized
        scores = np.array(
            [cosine_similarity(q_emb, doc_emb) for doc_emb in self.embeddings],
            dtype=np.float32,
        )

        # Get top-k indices (descending order)
        top_indices = np.argsort(scores)[-k:][::-1]

        return [SearchResult(self.documents[i], float(scores[i])) for i in top_indices]


# ──────────────────────────────────────────────────────────────────────────────
#  The rest of the file remains mostly unchanged
# ──────────────────────────────────────────────────────────────────────────────


class BM25Retriever:
    """BM25 retriever using rank_bm25"""

    def __init__(self):
        self.bm25: Optional[BM25Okapi] = None
        self.documents: List[Any] = []

    def index(self, documents: List[Any], texts: List[str]):
        if len(documents) != len(texts):
            raise ValueError("documents and texts must have same length")
        tokenized = [doc.lower().split() for doc in texts]
        self.bm25 = BM25Okapi(tokenized)
        self.documents.extend(documents)

    def search(self, query: str, k: int = 50) -> List[SearchResult]:
        if self.bm25 is None:
            return []
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[-k:][::-1]
        return [
            SearchResult(self.documents[i], float(scores[i]))
            for i in top_indices
            if scores[i] > 0
        ]


@dataclass
class HybridConfig:
    k_candidates: int = 80
    k_final: int = 20
    rrf_constant: float = 60.0
    bm25_weight: float = 1.0
    vector_weight: float = 1.0


def reciprocal_rank_fusion(
    result_lists: Sequence[List[SearchResult]],
    rrf_k: float = 60.0,
    weights: Optional[List[float]] = None,
) -> List[SearchResult]:
    """Merge ranked lists using Reciprocal Rank Fusion"""
    if not result_lists:
        return []
    if weights is None:
        weights = [1.0] * len(result_lists)
    if len(weights) != len(result_lists):
        raise ValueError("weights length must match number of lists")

    score_map: Dict[str, float] = {}
    id_to_doc: Dict[str, Any] = {}

    for ranked_list, weight in zip(result_lists, weights):
        for rank, res in enumerate(ranked_list, 1):
            doc_id = res.item["id"]
            rrf_score = weight / (rrf_k + rank)
            score_map[doc_id] = score_map.get(doc_id, 0.0) + rrf_score
            id_to_doc[doc_id] = res.item

    fused = [
        SearchResult(id_to_doc[doc_id], score)
        for doc_id, score in sorted(score_map.items(), key=lambda x: x[1], reverse=True)
    ]
    return fused


class HybridSearcher:
    @classmethod
    def from_documents(
        cls,
        documents: List[Dict[str, Any]],
        model: LLAMACPP_EMBED_KEYS | LlamacppEmbedding = "nomic-embed-text",
        base_url: Optional[str] = os.getenv("LLAMA_CPP_EMBED_URL"),
        query_prefix: Optional[str] = None,
        document_prefix: Optional[str] = None,
        use_cache: bool = True,
        embedder_kwargs: dict = {},
        **config_kwargs,
    ) -> "HybridSearcher":
        if not documents:
            raise ValueError("No documents provided")

        texts = [doc["content"] for doc in documents]
        for doc in documents:
            if not isinstance(doc, dict) or "id" not in doc or "content" not in doc:
                raise ValueError(
                    "Each document must be a dict with 'id' and 'content' keys"
                )

        if isinstance(model, str):
            embedder = LlamacppEmbedding(
                base_url=base_url,
                model=model,
                query_prefix=query_prefix,
                document_prefix=document_prefix,
                use_cache=use_cache,
                **embedder_kwargs,
            )
        else:
            embedder = model

        bm25_ret = BM25Retriever()
        vector_ret = VectorRetriever(embedder)

        bm25_ret.index(documents, texts)
        vector_ret.index(documents, texts)

        config = HybridConfig(**config_kwargs)
        return cls(bm25_ret, vector_ret, config)

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        vector_retriever: VectorRetriever,
        config: HybridConfig = HybridConfig(),
    ):
        self.bm25 = bm25_retriever
        self.vector = vector_retriever
        self.config = config

    def search(self, query: str) -> List[SearchResult]:
        bm25_results = self.bm25.search(query, k=self.config.k_candidates)
        vector_results = self.vector.search(query, k=self.config.k_candidates)

        fused = reciprocal_rank_fusion(
            [bm25_results, vector_results],
            rrf_k=self.config.rrf_constant,
            weights=[self.config.bm25_weight, self.config.vector_weight],
        )
        return fused[: self.config.k_final]


# ────────────────────────────────────────────────
#                   Example Usage
# ────────────────────────────────────────────────

if __name__ == "__main__":
    # Prepare data
    docs = [
        {
            "id": "d1",
            "content": "Hybrid vector search best practices 2025. Use RRF for combining BM25 and dense embeddings. Run both retrievers in parallel and fuse with reciprocal rank fusion...",
        },
        {
            "id": "d2",
            "content": "nomic-embed-text-v1.5 performance. Very fast on llama.cpp especially with Q5_K_M quantization. Low memory usage and excellent latency for local inference...",
        },
        {
            "id": "d3",
            "content": "Reciprocal Rank Fusion. Simple yet powerful fusion method used in Elastic, Weaviate, Azure Search, and many production RAG systems...",
        },
        {
            "id": "d4",
            "content": "Local embedding servers. llama.cpp provides OpenAI compatible API for embedding models like nomic-embed-text-v1.5. Easy to run on CPU/GPU...",
        },
        {
            "id": "d5",
            "content": "BM25 is still very strong. Especially good at rare terms, IDs, exact matches, product codes, and keyword precision in hybrid search...",
        },
    ]

    model: LLAMACPP_EMBED_KEYS = "nomic-embed-text"

    hybrid = HybridSearcher.from_documents(
        documents=docs,
        model=model,
        k_candidates=10,
        k_final=5,
        bm25_weight=1.2,
        vector_weight=1.0,
    )

    # Query
    query = "fast local embeddings with llama.cpp"
    results = hybrid.search(query)

    print(f"\nResults for: {query!r}\n")
    for i, res in enumerate(results, 1):
        doc = res.item
        preview = (
            doc["content"][:80] + "..." if len(doc["content"]) > 80 else doc["content"]
        )
        print(f"{i:2d}. {res.score:6.4f}  {doc['id']}  {preview}")
