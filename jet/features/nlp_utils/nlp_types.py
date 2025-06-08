from typing import Literal, Tuple, Union, List, Dict, Optional, TypedDict


class WordOccurrence(TypedDict):
    count: int
    start_idxs: List[int]
    end_idxs: List[int]
    sentence_idx: int
    word: str
    sentence: str


class NgramOccurrence(TypedDict):
    count: int
    start_idxs: List[int]
    end_idxs: List[int]
    sentence_idx: int
    sentence: str
    ngram: Tuple[str, ...]


class Matched(TypedDict):
    sentence_idx: int
    sentence: str
    ngrams: List[NgramOccurrence]


MatchedKeywords = Dict[str, List[WordOccurrence]]
