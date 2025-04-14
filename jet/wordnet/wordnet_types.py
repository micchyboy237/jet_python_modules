from typing import List, Optional, TypedDict


class SimilarityResult(TypedDict):
    id: Optional[str]
    rank: Optional[int]
    score: float
    text: str
    percent_difference: Optional[float]


class FilterResult(TypedDict):
    text: str
    score: float
    others: List[SimilarityResult]
