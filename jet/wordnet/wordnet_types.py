from typing import List, Optional, TypedDict


class SimilarityResult(TypedDict):
    id: Optional[str]
    rank: Optional[int]
    text: str
    score: float
    percent_difference: Optional[float]


class FilterResult(TypedDict):
    text: str
    score: float
    others: List[SimilarityResult]
