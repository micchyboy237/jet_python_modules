from typing import Optional, TypedDict


class JobEntities(TypedDict, total=False):
    technology_stack: list[str]
    role: list[str]
    application: list[str]
    qualifications: list[str]


class Entity(TypedDict):
    text: str
    label: str
    score: float


JobEntity = dict[str, list[str]]


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
    salary: Optional[str]
    job_type: Optional[str]
    hours_per_week: Optional[int]


class JobMetadata(TypedDict):
    id: str
    doc_id: str
    header_doc_id: str
    parent_id: Optional[str]
    doc_index: int
    chunk_index: int
    num_tokens: int
    header: str
    parent_header: Optional[str]
    content: str
    level: int
    parent_level: Optional[int]
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
