from typing import TypedDict

from shared.data_types.job_analytics import JobAnalytics
from shared.data_types.job_entities import JobEntities

JobAnalytics = JobAnalytics
JobEntities = JobEntities


class Entity(TypedDict):
    text: str
    label: str
    score: float


class JobData(TypedDict, total=False):
    """Base job data structure. Made total=False since many fields can be null/missing."""

    id: str
    link: str
    title: str
    company: str
    posted_date: str | None
    keywords: list[str] | None
    details: str | None
    entities: JobEntities | None
    tags: list[str] | None
    domain: str | None
    salary: str | None
    job_type: str | None
    hours_per_week: int | None


class ChunkMeta(TypedDict, total=False):
    """Metadata embedded within vector search result chunks."""

    level: int
    doc_id: str
    end_idx: int
    doc_index: int
    parent_id: str | None
    start_idx: int
    text_hash: str
    num_tokens: int
    chunk_index: int
    content_hash: str
    parent_level: int
    header_doc_id: str


class HybridMatchInfo(TypedDict, total=False):
    """BM25 keyword match counts from hybrid search reranking.

    Keys are dynamic lowercase query terms, values are match counts.
    Example: {"ai": 10, "llm": 5}
    """

    pass


class HybridResultMetadata(TypedDict, total=False):
    """Chunk-level metadata returned in hybrid search results (NOT full job metadata)."""

    parent_id: str | None
    doc_id: str
    chunk_index: int
    start_idx: int
    end_idx: int
    num_tokens: int
    parent_header: str
    header: str


class VectorSearchResult(TypedDict, total=False):
    """Result shape from search_jobs() with enriched metadata flattened at root level."""

    rank: int
    score: float
    id: str
    # Enriched job fields (flattened from metadata table)
    job_title: str
    company: str
    link: str
    keywords: list[str] | None
    entities: JobEntities | None
    domain: str | None
    salary: str | None
    job_type: str | None
    tags: list[str] | None
    hours_per_week: int | None
    # Chunk-level fields
    header: str
    parent_header: str
    content: str
    chunk_meta: ChunkMeta
    posted_date: str | None
    created_at: str | None
    updated_at: str | None


class HybridSearchResult(TypedDict, total=False):
    """Result shape from hybrid_search_jobs() with nested metadata and BM25 info."""

    rank: int
    id: str
    score: float
    similarity: float
    matched: HybridMatchInfo
    text: str
    metadata: HybridResultMetadata
    # Optionally enriched job fields (when enrich_with_metadata=True)
    job_title: str
    company: str
    link: str
    keywords: list[str] | None
    entities: JobEntities | None
    domain: str | None
    salary: str | None
    job_type: str | None
    tags: list[str] | None
    hours_per_week: int | None


# Backward-compatible alias pointing to the more accurate vector result type
JobSearchResult = VectorSearchResult


class JobChunk(TypedDict):
    id: str
    header: str
    content: str
    metadata: "JobChunkMetadata"
    embedding: list[float] | None


class JobChunkMetadata(TypedDict):
    job_id: str
    chunk_index: int
    start_idx: int
    end_idx: int
    num_tokens: int


class TableJobRow(TypedDict, total=False):
    id: str
    level: int
    doc_id: str
    header: str
    content: str
    end_idx: int
    metadata: "TableJobMetadata"
    doc_index: int
    embedding: list[float] | None
    parent_id: str
    start_idx: int
    text_hash: str
    created_at: str
    num_tokens: int
    updated_at: str
    chunk_index: int
    posted_date: str
    content_hash: str
    parent_level: int
    header_doc_id: str
    parent_header: str


class TableJobMetadata(TypedDict, total=False):
    id: str
    link: str
    tags: list[str]
    domain: str
    salary: str | None
    company: str
    entities: JobEntities
    job_type: str | None
    keywords: list[str]
    posted_date: str
    hours_per_week: int | None
