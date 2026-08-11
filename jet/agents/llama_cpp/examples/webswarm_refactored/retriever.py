import asyncio
import logging

import chromadb
from chromadb import EmbeddingFunction
from jet.adapters.llama_cpp.embed_utils import embed
from jet.adapters.llama_cpp.rerank_utils import rerank

from .config import DOC_CHAR_LIMIT, RERANK_TOP_K, VECTOR_DB_PATH

logger = logging.getLogger("webswarm")


class JetEmbeddingFunction(EmbeddingFunction):
    """Wraps jet.adapters.llama_cpp.embed_utils for ChromaDB.

    Prevents ChromaDB from downloading its default all-MiniLM-L6-v2 model
    by delegating all embedding calls to the project's existing embed utility.
    """

    def __call__(self, input: list[str]) -> list[list[float]]:
        result = embed(input, return_format="list", show_progress=False)
        if not isinstance(result, list):
            return [[] for _ in input]
        return result


class LocalRetriever:
    """Uses jet.adapters.llama_cpp embed/rerank utils + ChromaDB."""

    def __init__(self):
        self.chroma = chromadb.PersistentClient(path=VECTOR_DB_PATH)
        self.collection = self.chroma.get_or_create_collection(
            "swarm_findings",
            metadata={"hnsw:space": "cosine"},
            embedding_function=JetEmbeddingFunction(),
        )

    async def embed_texts(
        self, texts: list[str], max_retries: int = 3
    ) -> list[list[float]]:
        """Async wrapper around jet embed() with retry and VRAM safety."""
        loop = asyncio.get_running_loop()
        for attempt in range(max_retries):
            try:
                result = await loop.run_in_executor(
                    None,
                    lambda: embed(texts, return_format="list", show_progress=False),
                )
                if result and len(result) == len(texts):
                    return result
                logger.warning(
                    f"Embed attempt {attempt + 1}/{max_retries} returned "
                    f"incomplete results ({len(result) if result else 0}/{len(texts)})"
                )
            except Exception as e:
                logger.warning(f"Embed attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2**attempt)
        logger.error(f"All {max_retries} embed attempts failed for {len(texts)} texts")
        return []

    async def rerank_docs(
        self, query: str, docs: list[dict], top_k: int = RERANK_TOP_K
    ) -> list[dict]:
        """Async wrapper around jet rerank() using correct /rerank endpoint."""
        if not docs:
            return []
        doc_texts = [d.get("text", d.get("content", ""))[:DOC_CHAR_LIMIT] for d in docs]
        loop = asyncio.get_running_loop()
        rerank_results = await loop.run_in_executor(
            None, lambda: rerank(query, doc_texts, top_n=min(top_k, len(docs)))
        )
        return [docs[r["index"]] for r in rerank_results]

    async def store_finding(self, finding: dict):
        emb_list = await self.embed_texts([finding["content"][:DOC_CHAR_LIMIT]])

        if not emb_list or not emb_list[0]:
            logger.warning(
                f"Embedding failed for subtask '{finding['subtask_id']}'. "
                f"Storing without vector (text-only search will still work)."
            )
            self.collection.upsert(
                ids=[finding["subtask_id"]],
                documents=[finding["content"][:DOC_CHAR_LIMIT]],
                metadatas=[
                    {
                        "url": finding.get("url", ""),
                        "branch": finding.get("branch_id", ""),
                    }
                ],
            )
            return

        self.collection.upsert(
            ids=[finding["subtask_id"]],
            embeddings=[emb_list[0]],
            documents=[finding["content"][:DOC_CHAR_LIMIT]],
            metadatas=[
                {
                    "url": finding.get("url", ""),
                    "branch": finding.get("branch_id", ""),
                }
            ],
        )

    async def recall(
        self, query: str, top_k: int = 3, branch_filter: str | None = None
    ) -> list[dict]:
        where = {"branch": branch_filter} if branch_filter else None
        results = self.collection.query(
            query_texts=[query], n_results=top_k, where=where
        )
        if not results["documents"][0]:
            return []
        return [
            {"content": d, "url": m.get("url", ""), "score": s}
            for d, m, s in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]
