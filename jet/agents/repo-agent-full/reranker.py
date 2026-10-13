"""
Integration with BGE Reranker server for precision ranking.
Uses jet's production-grade reranking with BM25 fallback.
"""

from typing import Dict, List

from config import models
from jet.adapters.llama_cpp.rerank_utils import rerank as jet_rerank
from jet_telemetry import reranker
from utils import logger


class Reranker:
    @reranker(name="bge-reranker", model_name=models.RERANK_MODEL)
    async def rerank(self, query: str, documents: List[Dict]) -> List[Dict]:
        """
        Rerank documents using cross-encoder model with BM25 fallback.
        Args:
            query: Search query
            documents: List of document dicts with 'content' key
        Returns:
            Sorted list of documents by relevance score
        """
        if not documents:
            return []

        logger.info(f"Reranking {len(documents)} candidates...")
        texts = [doc["content"] for doc in documents]

        try:
            results = jet_rerank(
                query=query,
                documents=texts,
                top_n=len(documents),
                normalize_scores=True,
                sigmoid_temperature=1.0,
                method="auto",
                model=models.RERANK_MODEL,
            )

            if not results:
                logger.warning("Reranker returned no results")
                return documents

            # Map scores back to original documents
            for result in results:
                original_idx = result["index"]
                if 0 <= original_idx < len(documents):
                    documents[original_idx]["rerank_score"] = result["score"]
                    documents[original_idx]["raw_rerank_score"] = result["raw_score"]

            ranked_docs = sorted(
                documents, key=lambda x: x.get("rerank_score", 0), reverse=True
            )

            logger.info(
                f"Reranking complete. Top score: {ranked_docs[0].get('rerank_score', 0):.4f}"
            )
            return ranked_docs

        except Exception as e:
            logger.warning(f"Reranking failed, returning original order: {e}")
            return documents
