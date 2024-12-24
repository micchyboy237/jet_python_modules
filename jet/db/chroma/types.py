from pydantic import BaseModel
from typing import Callable, Optional, List, Any, TypedDict, Union


class VectorItem(BaseModel):
    id: str
    text: str
    vector: List[float | int]
    metadata: dict


class InitialDataEntry(TypedDict):
    id: str
    document: str
    embeddings: Optional[list[float]]
    metadata: Optional[dict[str, Union[str, int, float, bool]]]


class SearchResult(TypedDict):
    id: str
    document: str
    metadata: dict
    score: float
