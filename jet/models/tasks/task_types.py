from typing import TypedDict


class SimilarityResult(TypedDict):
    """
    Represents a single similarity result for a text.

    Fields:
        id: Identifier for the text. (Use uuid if ids are not provided)
        rank: Rank based on score (1 for highest).
        doc_index: Original index of the text in the input list.
        score: Normalized similarity score.
        text: The compared text (or chunk if long).
        tokens: Number of tokens from text.
    """
    id: str
    rank: int
    doc_index: int
    score: float
    text: str
    tokens: int


class RerankResult(TypedDict):
    """
    Represents a single reranked result for a text.

    Fields:
        id: Identifier for the text (same as in SimilarityResult).
        rank: Updated rank based on reranked score (1 for highest).
        doc_index: Original index of the text in the input list.
        score: Reranked similarity score (e.g., from cross-encoder or heuristic).
        text: The compared text (or chunk if long).
        tokens: Number of tokens from text.
    """
    id: str
    rank: int
    doc_index: int
    score: float
    text: str
    tokens: int
