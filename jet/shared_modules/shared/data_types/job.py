from typing import Any, TypedDict


class Entity(TypedDict):
    text: str
    label: str
    score: float


class JobEntities(TypedDict, total=False):
    company_name: list[str]
    job_location: list[str]
    salary_range: list[str]
    experience_level: list[str]
    employment_type: list[str]
    work_schedule: list[str]
    required_skills: list[str]
    used_technologies: list[str]
    programming_languages: list[str]
    key_responsibilities: list[str]
    requirements_qualifications: list[str]
    employee_benefits: list[str]
    technology_stack: list[str]


JobEntity = JobEntities


class JobData(TypedDict):
    id: str
    link: str
    title: str
    company: str
    posted_date: str
    keywords: list[str]
    details: str
    entities: JobEntities
    tags: list[str]
    domain: str
    salary: str | None
    job_type: str | None
    hours_per_week: int | None
    meta: "JobMetadata"


class JobMetadata(TypedDict):
    job_id: str
    chunk_index: int
    start_idx: int
    end_idx: int
    num_tokens: int
    # line_idx: int
    # overlap_start_idx: int | None
    # overlap_end_idx: int | None


class JobChunk(TypedDict):
    id: str
    header: str
    content: str
    metadata: JobMetadata
    embedding: list[float] | None


class JobSearchResultData(JobData):
    pass


class JobSearchResult(JobData):
    rank: int
    score: float
    id: str
    text: str
    metadata: JobSearchResultData


# Database row representations for jobs.json


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
    meta: dict[str, Any]
    tags: list[str]
    domain: str
    salary: str | None
    company: str
    entities: JobEntities
    job_type: str | None
    keywords: list[str]
    posted_date: str
    hours_per_week: int | None

    # These match the nested layout seen in the sample
    # of jobs.json under the "metadata" field
