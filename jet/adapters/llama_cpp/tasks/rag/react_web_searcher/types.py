"""Shared types for ReAct Web Searcher."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class QueryComplexity(str, Enum):
    SIMPLE = "simple"
    COMPLEX = "complex"


class QueryAnalysis(BaseModel):
    """Result of query classification and optional decomposition."""

    complexity: QueryComplexity = Field(
        description="Whether the query can be answered in one search or needs decomposition"
    )
    reasoning: str = Field(
        description="Brief explanation of why this complexity was chosen"
    )
    sub_queries: list[str] = Field(
        default_factory=list,
        description="2-5 focused sub-queries if complex, empty if simple",
    )
    refined_query: str = Field(
        description="Refined version of the original query for better search results",
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
    """One Thought → Action → Observation cycle in the ReAct loop."""

    thought: str
    action: str
    action_input: dict = field(default_factory=dict)
    observation: str = ""
    tokens_used: int = 0


@dataclass
class FinalAnswer:
    """Complete output from the ReAct web searcher."""

    answer: str
    sources: list[SearchResult] = field(default_factory=list)
    steps: list[AgentStep] = field(default_factory=list)
    confidence: str = "high"
    total_tokens: int = 0
    eval_result: Optional[dict] = None
