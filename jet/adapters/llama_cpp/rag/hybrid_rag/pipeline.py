import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

from .models import Document, RetrievalResult
from .reranker import CrossEncoderReranker
from .retrievers import BM25Retriever, DenseRetriever
from .rrf import reciprocal_rank_fusion

logger = logging.getLogger(__name__)


class HybridRAGPipeline:
    """
    Modern hybrid RAG pipeline:
    Dense + BM25 → RRF → Cross-Encoder → Final context
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        bm25_retriever: BM25Retriever,
        reranker: Optional[CrossEncoderReranker] = None,
        rrf_k: int = 60,
        candidate_k: int = 50,  # how many to fetch from each retriever
        final_k: int = 8,  # final chunks for LLM
        weights: Optional[Dict[str, float]] = None,
    ):
        self.dense = dense_retriever
        self.bm25 = bm25_retriever
        self.reranker = reranker or CrossEncoderReranker()
        self.rrf_k = rrf_k
        self.candidate_k = candidate_k
        self.final_k = final_k
        self.weights = weights or {"dense": 1.0, "bm25": 1.0}

        logger.info(
            f"HybridRAGPipeline ready | candidate_k={candidate_k} "
            f"final_k={final_k} rrf_k={rrf_k}"
        )

    def retrieve(self, query: str) -> List[Document]:
        """
        Full retrieval pipeline:
        1. Parallel dense + BM25
        2. RRF fusion
        3. Cross-encoder re-ranking
        """
        logger.info(f"Starting hybrid retrieval for: {query[:80]}...")

        # 1. Parallel retrieval
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(self.dense.retrieve, query, self.candidate_k): "dense",
                executor.submit(self.bm25.retrieve, query, self.candidate_k): "bm25",
            }

            results: List[RetrievalResult] = []
            for future in as_completed(futures):
                results.append(future.result())

        # 2. RRF fusion
        fused = reciprocal_rank_fusion(
            results=results,
            k=self.rrf_k,
            weights=self.weights,
            top_n=self.candidate_k,  # keep a good pool for re-ranking
        )

        # 3. Precision re-ranking
        final_docs = self.reranker.rerank(query, fused, top_k=self.final_k)

        logger.info(f"Pipeline finished → {len(final_docs)} final documents")
        return final_docs

    def format_context(self, documents: List[Document]) -> str:
        """Simple context formatter for the LLM."""
        parts = []
        for i, doc in enumerate(documents, 1):
            parts.append(
                f"[{i}] (rrf={doc.rrf_score:.4f}, rerank={doc.rerank_score:.4f})\n"
                f"{doc.content}"
            )
        return "\n\n".join(parts)
