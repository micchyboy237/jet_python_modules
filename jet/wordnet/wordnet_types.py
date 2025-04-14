from typing import List, Optional, TypedDict


class SimilarityResult(TypedDict):
    """
    Represents a single similarity result for a text.

    Fields:
        id: Optional identifier for the text.
        rank: Optional rank based on score (1 for highest score).
        text: The compared text.
        score: Similarity score between query and text.
        percent_difference: Percentage difference from the highest score, rounded to 2 decimals.
    """
    id: Optional[str]
    rank: Optional[int]
    score: float
    percent_difference: Optional[float]
    text: str


class FilterResult(TypedDict):
    text: str
    score: float
    others: List[SimilarityResult]
