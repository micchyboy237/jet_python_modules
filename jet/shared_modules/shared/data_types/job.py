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
    entities: JobEntity
    tags: list[str]
    domain: str
    salary: Optional[str]
    job_type: Optional[str]
    hours_per_week: Optional[int]
