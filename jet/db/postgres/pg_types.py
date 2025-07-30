import numpy as np
from typing import Any, List, Dict, Optional, Tuple, TypedDict, Union
from numpy.typing import NDArray

Embedding = NDArray[np.float64]
EmbeddingInput = Union[List[float], Embedding]


class DatabaseMetadata(TypedDict):
    dbname: str
    owner: str
    encoding: str
    collation: Optional[str]
    ctype: Optional[str]
    size_mb: float


class ColumnMetadata(TypedDict):
    column_name: str
    data_type: str
    is_nullable: str
    character_maximum_length: Optional[int]
    numeric_precision: Optional[int]
    numeric_scale: Optional[int]


class TableRow(TypedDict):
    id: str
    # Allow any additional columns with arbitrary key-value pairs
    __annotations__: Dict[str, Any]


class SearchResult(TableRow):
    rank: int
    score: float


class TableMetadata(TypedDict):
    table_name: str
    table_type: str
    schema_name: str
    row_count: int
    columns: List[ColumnMetadata]
