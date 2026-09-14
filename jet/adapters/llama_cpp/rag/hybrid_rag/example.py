"""
Example: Hybrid Retrieval-Augmented Generation (RAG) Pipeline

Minimal demonstration of a hybrid RAG pipeline using both dense and BM25 retrievers fused by Reciprocal Rank Fusion (RRF).

Usage:
    python -m jet.adapters.llama_cpp.rag.hybrid_rag.example
"""

import logging

from .models import Document
from .pipeline import HybridRAGPipeline
from .retrievers import (
    BM25Retriever,
    DenseRetriever,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    # -------------------------------------------------
    # 1. Sample knowledge base
    # -------------------------------------------------
    sample_docs = [
        Document(
            id="d1",
            content="Reciprocal Rank Fusion (RRF) combines ranked lists from multiple retrievers by summing 1/(k + rank). The constant k is usually 60.",
        ),
        Document(
            id="d2",
            content="Hybrid search in RAG systems runs dense vector search and BM25 keyword search in parallel, then fuses the results.",
        ),
        Document(
            id="d3",
            content="A cross-encoder re-ranker scores query-document pairs jointly and produces much higher precision than bi-encoders.",
        ),
        Document(
            id="d4",
            content="BM25 is a probabilistic ranking function that works especially well for exact keyword matching and rare terms.",
        ),
        Document(
            id="d5",
            content="Dense retrievers using sentence transformers capture semantic similarity even when the wording is different.",
        ),
        Document(
            id="d6",
            content="In modern RAG pipelines the recommended pattern is hybrid retrieval with RRF followed by a cross-encoder re-ranker.",
        ),
        Document(
            id="d7",
            content="The original RRF paper was published in 2009 by Cormack, Clarke and Buettcher.",
        ),
        Document(
            id="d8",
            content="Fetching a larger candidate pool (50-100) from each retriever before fusion gives RRF more room to promote consensus documents.",
        ),
    ]

    # -------------------------------------------------
    # 2. Build retrievers
    # -------------------------------------------------
    dense = DenseRetriever(documents=sample_docs)
    bm25 = BM25Retriever(documents=sample_docs)

    # -------------------------------------------------
    # 3. Create pipeline
    # -------------------------------------------------
    pipeline = HybridRAGPipeline(
        dense_retriever=dense,
        bm25_retriever=bm25,
        candidate_k=6,  # small for demo
        final_k=4,
        rrf_k=60,
        weights={"dense": 1.0, "bm25": 1.0},
    )

    # -------------------------------------------------
    # 4. Run a query
    # -------------------------------------------------
    query = "How does reciprocal rank fusion work in hybrid RAG?"

    final_docs = pipeline.retrieve(query)

    print("\n" + "=" * 70)
    print(f"QUERY: {query}")
    print("=" * 70)
    print(pipeline.format_context(final_docs))
    print("=" * 70)


if __name__ == "__main__":
    main()
