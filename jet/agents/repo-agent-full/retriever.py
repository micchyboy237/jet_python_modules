"""
Hybrid retrieval logic combining vector search and reranking.
"""

from typing import Dict, List, Optional

from config import retrieval
from jet_telemetry import retriever
from reranker import Reranker
from utils import logger
from vector_store import VectorStore


class Retriever:
    def __init__(self):
        self.store = VectorStore()
        self.reranker = Reranker()

    @retriever(name="hybrid-retriever", model_name="lancedb+bge")
    async def retrieve(self, query: str, filters: Optional[Dict] = None) -> List[Dict]:
        if not query or not query.strip():
            logger.warning("Empty query provided")
            return []

        logger.info(f"Retrieving context for: '{query[:50]}...'")

        # Vector Search
        candidates = await self.store.search(query, k=retrieval.TOP_K_INITIAL)

        valid_candidates = [
            c for c in candidates if c["_distance"] <= retrieval.DISTANCE_THRESHOLD
        ]

        if not valid_candidates:
            logger.warning("No relevant chunks found within distance threshold.")
            return []

        logger.info(f"Found {len(valid_candidates)} candidates within threshold")

        # Reranking
        ranked_docs = await self.reranker.rerank(query, valid_candidates)

        final_results = ranked_docs[: retrieval.TOP_K_FINAL]
        logger.info(f"Returning top {len(final_results)} results")

        return final_results
