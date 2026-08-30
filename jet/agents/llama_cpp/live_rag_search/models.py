from dataclasses import dataclass, field
from enum import Enum


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
