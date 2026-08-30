# jet_python_modules/jet/agents/llama_cpp/live_rag_search/models.py

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List

from pydantic import BaseModel, Field

# --- Existing Dataclasses ---


class SufficiencyStatus(Enum):
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    ERROR = "error"


@dataclass
class SufficiencyResult:
    status: SufficiencyStatus
    missing_fields: list[str] = field(default_factory=list)
    reasoning: str = ""


@dataclass
class SearchResult:
    url: str
    snippet: str
    score: float
    title: str = ""


@dataclass
class ExtractedFacts:
    """Keyed by canonical entity ID (e.g., anime title normalized)."""

    entities: dict[str, dict] = field(default_factory=dict)


# --- New Pydantic Models for Structured Output ---


class SufficiencyResponse(BaseModel):
    """Schema for sufficiency evaluation."""

    status: str = Field(..., description="Must be 'complete' or 'incomplete'")
    reasoning: str = Field(..., description="Brief explanation of why")


class ExtractionResponse(BaseModel):
    """Schema for fact extraction. Enforces nested dict structure."""

    entities: Dict[str, Dict[str, str]] = Field(
        ...,
        description="Dictionary where keys are entity names and values are dicts of facts",
    )


class LinkFilterResponse(BaseModel):
    """Schema for link filtering."""

    urls: List[str] = Field(..., description="List of relevant absolute URLs")
