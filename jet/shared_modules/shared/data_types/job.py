from typing import Any, TypedDict


class JobEntities(TypedDict, total=False):
    technology_stack: list[str]
    role: list[str]
    application: list[str]
    qualifications: list[str]


class Entity(TypedDict):
    text: str
    label: str
    score: float


JobEntity = dict[str, Any]


class JobData(TypedDict):
    id: str
    link: str
    title: str
    company: str
    posted_date: str
    keywords: list[str]
    details: str
    # entities: JobEntity
    entities: JobEntities
    tags: list[str]
    domain: str
    salary: str | None
    job_type: str | None
    hours_per_week: int | None


class JobMetadata(TypedDict):
    id: str
    doc_id: str
    header_doc_id: str
    parent_id: str | None
    doc_index: int
    chunk_index: int
    num_tokens: int
    header: str
    parent_header: str | None
    content: str
    level: int
    parent_level: int | None
    start_idx: int
    end_idx: int


class JobSearchResultData(JobData):
    pass


class JobSearchResult(JobData):
    rank: int
    score: float
    id: str
    text: str
    metadata: JobSearchResultData
