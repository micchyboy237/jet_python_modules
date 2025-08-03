from typing import List, Optional, TypedDict, Tuple, Union

# Define primitive types for category_values
PrimitiveType = Union[str, int, float, bool]


class ProcessedData(TypedDict):
    source: str
    target: Optional[str]
    category_values: List[PrimitiveType]
    score: Optional[float]


class ProcessedDataString(TypedDict):
    source: str
    category_values: List[PrimitiveType]


class StratifiedData(TypedDict):
    source: str
    target: Optional[str]
    score: Optional[float]
    category_values: List[PrimitiveType]
