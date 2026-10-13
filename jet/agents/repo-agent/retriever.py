"""
Hybrid retrieval logic combining vector search and reranking.
"""

from typing import Dict, List, Optional

from config import retrieval
from reranker import Reranker
from utils import logger
from vector_store import VectorStore


class Retriever:
    def __init__(self):
        self.store = VectorStore()
        self.reranker = Reranker()

    async def retrieve(self, query: str, filters: Optional[Dict] = None) -> List[Dict]:
        logger.info(f"Retrieving context for: '{query[:50]}...'")

        # Stage 1: Vector Search
        candidates = await self.store.search(query, k=retrieval.TOP_K_INITIAL)

        # Filter by distance threshold
        valid_candidates = [
            c for c in candidates if c["_distance"] <= retrieval.DISTANCE_THRESHOLD
        ]

        if not valid_candidates:
            logger.warning("No relevant chunks found within distance threshold.")
            return []

        # Stage 2: Reranking
        ranked_docs = await self.reranker.rerank(query, valid_candidates)

        # Return top K final
        return ranked_docs[: retrieval.TOP_K_FINAL]
