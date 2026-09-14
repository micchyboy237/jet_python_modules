from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Document:
    """A single retrieved chunk / document."""

    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0  # Raw retriever score (optional)
    rrf_score: float = 0.0  # After RRF
    rerank_score: float = 0.0  # After cross-encoder


@dataclass
class RetrievalResult:
    """Result from one retriever."""

    retriever_name: str
    documents: List[Document]
