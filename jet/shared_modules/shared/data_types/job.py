from typing import List, Optional, TypedDict, Union

from pydantic import BaseModel, Field, field_validator


class Entity(TypedDict):
    text: str
    label: str
    score: float


class JobEntities(BaseModel):
    job_title: str = Field(description="Official job title/position name")
    key_responsibilities: List[str] = Field(
        description="Main duties and responsibilities of the role"
    )
    company_name: Optional[str] = Field(None, description="Name of the hiring company")
    company_summary: Optional[str] = Field(
        None, description="Brief overview of what the company does"
    )
    nature_of_business: Optional[str] = Field(
        None,
        description="Industry sector and type of business, e.g., 'SaaS', 'Healthcare', 'E-commerce'",
    )
    company_size: Optional[str] = Field(
        None,
        description="Approximate company size, e.g., 'Startup', '50-200 employees', 'Enterprise'",
    )
    department: Optional[str] = Field(
        None, description="Department or team within the company"
    )
    job_description_summary: Optional[str] = Field(
        None, description="2-3 sentence overview of the role and its purpose"
    )
    job_location: Optional[str] = Field(
        None, description="Primary work location - city, state/country"
    )
    remote_work_policy: Optional[str] = Field(
        None,
        description="e.g., 'Fully Remote', 'Hybrid (2 days/week in office)', 'On-site'",
    )
    salary_range: Optional[str] = Field(
        None,
        description="Salary range or compensation information, e.g., '$80K-$120K', 'Competitive'",
    )
    employment_type: Optional[str] = Field(
        None,
        description="e.g., 'Full-time', 'Part-time', 'Contract', 'Freelance', 'Internship'",
    )
    experience_level: Optional[str] = Field(
        None,
        description="e.g., 'Entry Level', 'Mid-Level', 'Senior', 'Lead', 'Manager', 'Director'",
    )
    schedule_type: Optional[str] = Field(
        None,
        description="e.g., 'Flexible hours', '9-to-5', 'Shift work', 'Weekend availability'",
    )
    required_skills: Optional[List[str]] = Field(
        None, description="Must-have technical and soft skills"
    )
    preferred_skills: Optional[List[str]] = Field(
        None, description="Nice-to-have but not required skills"
    )
    technology_stack: Optional[List[str]] = Field(
        None, description="Specific technologies, tools, frameworks mentioned"
    )
    years_of_experience: Optional[str] = Field(
        None,
        description="Required years of experience, e.g., '3+ years', '5-7 years'",
    )
    education_requirements: Optional[Union[str, List[str]]] = Field(
        None, description="Required degrees or educational background"
    )
    certifications_required: Optional[Union[str, List[str]]] = Field(
        None, description="Required certifications or licenses"
    )
    language_requirements: Optional[Union[str, List[str]]] = Field(
        None, description="Required languages for the role"
    )
    qualifications: Optional[Union[str, List[str]]] = Field(
        None,
        description="Combined qualifications, experience, and competency requirements",
    )
    employee_benefits: Optional[Union[str, List[str]]] = Field(
        None, description="Benefits, perks, and compensation extras offered"
    )
    application_instructions: Optional[str] = Field(
        None,
        description="How to apply, including any links, emails, or special instructions",
    )
    application_deadline: Optional[str] = Field(
        None, description="Application deadline if specified"
    )

    @field_validator(
        "education_requirements",
        "certifications_required",
        "language_requirements",
        "qualifications",
        "employee_benefits",
        mode="before",
    )
    @classmethod
    def ensure_list(cls, v):
        """Convert string values to lists or None if they indicate no data"""
        if v is None:
            return None
        if isinstance(v, str):
            if v.lower() in [
                "none specified",
                "not mentioned",
                "none",
                "n/a",
                "not specified",
            ]:
                return None
            return [v]
        return v


JobEntity = JobEntities


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
