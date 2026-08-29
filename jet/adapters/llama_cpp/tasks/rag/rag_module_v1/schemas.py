# rag_module_v1/schemas.py

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SearchStatus(str, Enum):
    FOUND = "found"
    ABSTAINED = "abstained"
    ERROR = "error"


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    vector_score: float | None = None
    bm25_score: float | None = None
    rerank_score: float | None = None
    arms: list[str] = field(default_factory=list)


@dataclass
class SearchResult:
    status: SearchStatus
    answer_context: str = ""
    sources: list[dict[str, Any]] = field(default_factory=list)
    query_used: str = ""
    metadata_applied: dict[str, Any] = field(default_factory=dict)
    truncated: bool = False
    _latency_ms: int = 0

    def to_dict(self, include_internal: bool = True) -> dict:
        d = {
            "status": self.status.value,
            "answer_context": self.answer_context,
            "sources": self.sources,
            "query_used": self.query_used,
            "metadata_applied": self.metadata_applied,
            "truncated": self.truncated,
            "_latency_ms": self._latency_ms,
        }
        if not include_internal:
            d.pop("_latency_ms", None)
        return d
