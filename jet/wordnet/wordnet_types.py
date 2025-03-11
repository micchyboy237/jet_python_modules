from typing import List, TypedDict


class SimilarityResult(TypedDict):
    text: str
    score: float
    percent_difference: float


class FilterResult(TypedDict):
    text: str
    score: float
    others: List[SimilarityResult]
