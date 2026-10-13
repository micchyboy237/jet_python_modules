"""
Integration with BGE Reranker server for precision ranking.
"""

from typing import Dict, List

import aiohttp
from config import models, servers
from utils import logger


class Reranker:
    async def rerank(self, query: str, documents: List[Dict]) -> List[Dict]:
        if not documents:
            return []

        logger.info(f"Reranking {len(documents)} candidates...")

        # Prepare pairs for BGE API: [[query, doc], [query, doc]]
        pairs = [[query, doc["content"]] for doc in documents]

        try:
            async with aiohttp.ClientSession() as session:
                payload = {"pairs": pairs, "model": models.RERANK_MODEL}
                async with session.post(
                    f"{servers.RERANK_URL}/rerank", json=payload
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        scores = [item["score"] for item in data["results"]]

                        # Attach scores to docs
                        for i, doc in enumerate(documents):
                            doc["rerank_score"] = scores[i]

                        # Sort by score descending
                        return sorted(
                            documents, key=lambda x: x["rerank_score"], reverse=True
                        )
                    else:
                        logger.error(f"Reranker API error: {resp.status}")
                        return documents
        except Exception as e:
            logger.warning(f"Reranking failed, using original order: {e}")
            return documents
