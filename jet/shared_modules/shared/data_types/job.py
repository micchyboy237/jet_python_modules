from typing import Any, TypedDict


class Entity(TypedDict):
    text: str
    label: str
    score: float


JobEntity = dict[str, Any]


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


class JobMetadata(TypedDict):
    job_id: str
    chunk_index: int
    num_tokens: int
    start_idx: int
    end_idx: int
    line_idx: int
    overlap_start_idx: int | None
    overlap_end_idx: int | None


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
