"""Shared types for ReAct Web Searcher."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class QueryIntent(str, Enum):
    """Classifies the structural intent of a query for search strategy routing."""

    LIST = "list"  # Top N, best of, rankings, recommendations
    COMPARISON = "comparison"  # X vs Y, pros/cons, alternatives
    FACTUAL = "factual"  # Single fact, definition, date, entity lookup
    COMPLEX = "complex"  # Multi-faceted, requires decomposition
    UNKNOWN = "unknown"  # Fallback


class QueryComplexity(str, Enum):
    SIMPLE = "simple"
    COMPLEX = "complex"


class QueryAnalysis(BaseModel):
    """Result of query classification and optional decomposition."""

    complexity: QueryComplexity = Field(
        description="Whether the query can be answered in one search or needs decomposition"
    )
    intent: QueryIntent = Field(
        default=QueryIntent.UNKNOWN,
        description=(
            "Structural intent of the query. 'list' for rankings/top-N/recommendations, "
            "'comparison' for X-vs-Y, 'factual' for single facts, 'complex' for multi-faceted. "
            "This determines the search strategy used by the ReAct engine."
        ),
    )
    reasoning: str = Field(
        description="Brief explanation of why this complexity and intent were chosen"
    )
    sub_queries: list[str] = Field(
        default_factory=list,
        description="2-5 focused sub-queries if complex, empty otherwise",
    )
    refined_query: str = Field(
        description=(
            "Refined version of the original query for better search results. "
            "For list/ranking queries, preserve temporal constraints (e.g., '2026') "
            "and add 'list' or 'ranking' keywords. Do NOT remove year/specifiers."
        ),
    )


@dataclass
class SearchResult:
    """A single result from SearXNG."""

    title: str
    url: str
    snippet: str
    engine: str
    score: float = 0.0


@dataclass
class AgentStep:
    """One Thought → Action → Observation cycle in the ReAct loop.

    ✅ IMPROVEMENT: Added source_url and source_title fields so tools
    can propagate provenance metadata for citation in FinalAnswer.
    """

    thought: str
    action: str
    action_input: dict = field(default_factory=dict)
    observation: str = ""
    tokens_used: int = 0
    source_url: str | None = None  # ← NEW: URL of the source read/searched
    source_title: str | None = None  # ← NEW: Title of the source page


@dataclass
class FinalAnswer:
    """Complete output from the ReAct web searcher."""

    answer: str
    sources: list[SearchResult] = field(default_factory=list)
    steps: list[AgentStep] = field(default_factory=list)
    confidence: str = "high"
    total_tokens: int = 0
    truncated: bool = False
    eval_result: Optional[dict] = None
