"""Pydantic models for structured LLM outputs in the deep research pipeline."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SubQuery(BaseModel):
    """A single decomposed sub-query with optional dependency."""

    id: str = Field(description="Unique identifier, e.g. 'sq_1'")
    text: str = Field(description="Self-contained rewritten sub-query")
    depends_on: str | None = Field(
        default=None,
        description="ID of sub-query this depends on, or null if independent",
    )


class QueryPlan(BaseModel):
    """Output of the query planning step: rewrite + conditional decomposition."""

    needs_decomposition: bool = Field(
        description="True if query requires multi-step decomposition"
    )
    rewritten_query: str = Field(
        description="Rewritten/clarified version of the original query"
    )
    sub_queries: list[SubQuery] = Field(
        default_factory=list,
        description="Decomposed sub-queries (empty if needs_decomposition=False)",
    )


class GroundingResult(BaseModel):
    """Validation result for whether page content answers a sub-query."""

    is_grounded: bool = Field(
        description="True if content contains verifiable evidence for the sub-query"
    )
    evidence_summary: str = Field(
        description="Brief summary of the evidence found, or reason for failure"
    )
    relevant_section: str = Field(
        default="",
        description="Exact quote or section from content that supports the answer",
    )


class ExtractedLink(BaseModel):
    """A link extracted from page content that may answer an unanswered sub-query."""

    url: str = Field(description="Absolute URL of the link")
    anchor_text: str = Field(description="Anchor text of the link")
    target_sub_query_id: str | None = Field(
        default=None,
        description="Which unanswered sub-query this link might address",
    )


class ExtractedLinks(BaseModel):
    """Collection of context-aware extracted links."""

    links: list[ExtractedLink] = Field(default_factory=list)


class Citation(BaseModel):
    """A verified citation bound to a specific claim."""

    claim: str = Field(description="The claim being cited")
    source_url: str = Field(description="URL where evidence was found")
    evidence_quote: str = Field(description="Exact quote supporting the claim")
    section_header: str = Field(default="", description="Section header for context")


class SynthesizedAnswer(BaseModel):
    """Final synthesized answer with citations and uncertainty flags."""

    answer: str = Field(description="Complete answer to the original query")
    citations: list[Citation] = Field(default_factory=list)
    unresolved_sub_queries: list[str] = Field(
        default_factory=list,
        description="Sub-queries that could not be fully answered",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall confidence score (0.0 = no evidence, 1.0 = fully verified)",
    )
