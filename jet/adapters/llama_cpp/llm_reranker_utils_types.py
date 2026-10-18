from typing import Any, TypedDict


class RankingResult(TypedDict):
    """Typed dictionary for a single ranked document with explanation."""

    index: int
    score: float
    reason: str
    metadata: dict[str, Any]
    text: str


class CompactRankingResult(TypedDict):
    """Typed dictionary for minimal ranking result (token-efficient)."""

    index: int
    score: float
    metadata: dict[str, Any]
    text: str
