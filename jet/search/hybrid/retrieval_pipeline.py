from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer


@dataclass
class Document:
    id: str
    text: str
    metadata: dict[str, Any]


class InMemoryVectorStore:
    def __init__(self, embedding_model: SentenceTransformer):
        self.embedding_model = embedding_model
        self.documents: list[Document] = []
        self.embeddings: np.ndarray | None = None

    def add_documents(self, docs: list[Document]) -> None:
        self.documents.extend(docs)
        texts = [d.text for d in self.documents]
        self.embeddings = self.embedding_model.encode(texts, normalize_embeddings=True)

    def metadata_filter(self, filters: dict[str, Any]) -> list[int]:
        indices = []
        for idx, doc in enumerate(self.documents):
            match = all(doc.metadata.get(k) == v for k, v in filters.items())
            if match:
                indices.append(idx)
        return indices

    def vector_search(self, query: str, indices: list[int], top_k: int = 100):
        query_emb = self.embedding_model.encode([query], normalize_embeddings=True)[0]
        subset_emb = self.embeddings[indices]
        scores = np.dot(subset_emb, query_emb)
        ranked = sorted(zip(indices, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


class BM25Retriever:
    def __init__(self, docs: list[Document]):
        self.docs = docs
        tokenized = [d.text.split() for d in docs]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 100):
        tokens = query.split()
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


def reciprocal_rank_fusion(
    vector_results: list[tuple], bm25_results: list[tuple], k: int = 60
):
    fused_scores = {}

    for rank, (idx, _) in enumerate(vector_results):
        fused_scores[idx] = fused_scores.get(idx, 0) + 1 / (k + rank + 1)

    for rank, (idx, _) in enumerate(bm25_results):
        fused_scores[idx] = fused_scores.get(idx, 0) + 1 / (k + rank + 1)

    ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked


class RetrievalPipeline:
    def __init__(self):
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.vector_store = InMemoryVectorStore(self.embedding_model)
        self.documents: list[Document] = []
        self.bm25: BM25Retriever | None = None

    def add_documents(self, docs: list[Document]) -> None:
        self.documents = docs
        self.vector_store.add_documents(docs)
        self.bm25 = BM25Retriever(docs)

    def retrieve(
        self, query: str, metadata_filters: dict[str, Any] | None = None, top_k: int = 5
    ) -> list[Document]:
        metadata_filters = metadata_filters or {}

        # Metadata filtering
        indices = self.vector_store.metadata_filter(metadata_filters)

        # Vector search
        vector_results = self.vector_store.vector_search(query, indices, top_k=100)

        # BM25 search
        bm25_results = self.bm25.search(query, top_k=100)

        # Filter BM25 results to only include documents that pass metadata filter
        filtered_bm25 = [(idx, score) for idx, score in bm25_results if idx in indices]
        bm25_results = filtered_bm25[:100]  # keep same budget

        # RRF fusion
        fused = reciprocal_rank_fusion(vector_results, bm25_results)

        # Top 50 for reranking
        top_candidates = [idx for idx, _ in fused[:50]]

        pairs = [(query, self.documents[idx].text) for idx in top_candidates]
        scores = self.reranker.predict(pairs)

        reranked = sorted(zip(top_candidates, scores), key=lambda x: x[1], reverse=True)

        final_indices = [idx for idx, _ in reranked[:top_k]]

        return [self.documents[i] for i in final_indices]
