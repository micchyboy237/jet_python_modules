from typing import Optional, TypedDict, List, Dict, Union


class Match(TypedDict):
    text: str
    start_idx: int
    end_idx: int


class Metadata(TypedDict, total=False):
    query_scores: Dict[str, float]


class SearchResult(TypedDict):
    rank: int
    score: float
    header: str
    content: str
    id: str
    metadata: Metadata
    matches: List[Match]
