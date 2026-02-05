import uuid
from dataclasses import dataclass
from typing import Any

from jet.adapters.llama_cpp.hybrid_search import (
    RELATIVE_CATEGORY_CONFIG,
    HybridSearch,
)
from jet.adapters.llama_cpp.types import LLAMACPP_EMBED_KEYS


@dataclass
class ContentChunk:
    id: str
    content: str
    source_url: str | None
    added_at_step: int
    relevance_score: float | None = None


class LongTermContentMemory:
    """In-memory hybrid (dense + sparse) store for accumulated research content"""

    def __init__(
        self,
        embed_model: LLAMACPP_EMBED_KEYS = "nomic-embed-text",
        max_chunks: int = 1500,
    ):
        self.embed_model = embed_model
        self.max_chunks = max_chunks
        self.chunks: list[ContentChunk] = []
        self._hybrid: HybridSearch | None = None
        self._rebuild_needed = True

    def add(
        self,
        content: str,
        source_url: str | None = None,
        step: int = 0,
    ) -> str:
        """Add a piece of relevant content to long-term memory"""
        if not content or not content.strip():
            return ""

        chunk_id = str(uuid.uuid4())[:10]
        chunk = ContentChunk(
            id=chunk_id,
            content=content.strip(),
            source_url=source_url,
            added_at_step=step,
        )
        self.chunks.append(chunk)
        self._rebuild_needed = True

        if len(self.chunks) > self.max_chunks:
            self.chunks = self.chunks[-self.max_chunks :]

        return chunk_id

    def _ensure_index(self) -> None:
        if not self._rebuild_needed or not self.chunks:
            return

        documents = [c.content for c in self.chunks]
        ids = [c.id for c in self.chunks]

        self._hybrid = HybridSearch.from_documents(
            documents=documents,
            ids=ids,
            model=self.embed_model,
            dense_weight=1.4,
            sparse_weight=0.8,
            category_config=RELATIVE_CATEGORY_CONFIG,
        )
        self._rebuild_needed = False

    def search(
        self,
        query: str,
        top_k: int = 7,
        min_score: float = 0.20,
    ) -> list[dict[str, Any]]:
        self._ensure_index()
        if not self._hybrid:
            return []

        raw_results = self._hybrid.search(
            query,
            top_k=top_k * 2,
            normalize_scores=True,
        )

        filtered = []
        for r in raw_results:
            if r["hybrid_score"] < min_score:
                continue
            chunk = next((c for c in self.chunks if c.id == r["id"]), None)
            if chunk:
                filtered.append(
                    {
                        "id": chunk.id,
                        "content": chunk.content,
                        "source": chunk.source_url or "[unknown source]",
                        "score": r["hybrid_score"],
                        "step": chunk.added_at_step,
                    }
                )

        filtered.sort(key=lambda x: x["score"], reverse=True)
        return filtered[:top_k]
