# jet_python_modules/jet/db/postgres/pg_types.py
from typing import Any, Dict, List, Optional, TypedDict, Union

import numpy as np
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


class TableRow(TypedDict, total=False):
    id: str
    __annotations__: Dict[str, Any]


class SearchResult(TableRow):
    """
    Base type for all search results.
    Ensures every result has a rank and a primary scoring metric.
    """

    rank: int
    score: float


class SemanticSearchResult(SearchResult):
    """Specific to vector-only search."""

    distance: float


class HybridSearchResult(SearchResult):
    """Specific to Hybrid (Vector + Keyword) search."""

    hybrid_score: float


class DiverseSearchResult(SearchResult):
    """Specific to MMR/Diversity search."""

    pass


class MultiQueryResult(SearchResult):
    """Specific to Multi-Query expansion search."""

    rrf_score: float


class TableMetadata(TypedDict):
    table_name: str
    table_type: str
    schema_name: str
    row_count: int
    columns: List[ColumnMetadata]
