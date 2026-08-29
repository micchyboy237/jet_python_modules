"""RAG Module v1 - Eval-driven agentic knowledge search tool.

Public API:
    search_knowledge(query, thought_context="") -> dict
    KnowledgeSearchTool - Stateful tool class for custom corpus/config
    RAGConfig - Frozen configuration dataclass
    SearchResult, SearchStatus - Output types
"""

from .config import RAGConfig
from .schemas import SearchResult, SearchStatus
from .search_knowledge import KnowledgeSearchTool, search_knowledge

__all__ = [
    "search_knowledge",
    "KnowledgeSearchTool",
    "RAGConfig",
    "SearchResult",
    "SearchStatus",
]
