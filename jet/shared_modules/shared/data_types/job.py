from typing import TypedDict


class JobEntities(TypedDict, total=False):
    technology_stack: list[str]
    role: list[str]
    application: list[str]
    qualifications: list[str]


class JobData(TypedDict):
    id: str
    link: str
    title: str
    company: str
    posted_date: str
    keywords: list[str]
    domain: str
    salary: str
    job_type: str
    hours_per_week: int
    tags: list[str]
    details: str
    entities: JobEntities
