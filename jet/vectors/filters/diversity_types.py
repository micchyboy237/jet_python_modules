from typing import TypedDict


class DiverseResult(TypedDict):
    id: str
    index: int
    text: str
    score: float


__all__ = ["DiverseResult"]
