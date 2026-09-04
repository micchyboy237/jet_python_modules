from datetime import date
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class EmploymentType(str, Enum):
    FULL_TIME = "full_time"
    PART_TIME = "part_time"
    CONTRACT = "contract"
    INTERNSHIP = "internship"


class WorkMode(str, Enum):
    REMOTE = "remote"
    ONSITE = "onsite"
    HYBRID = "hybrid"


class JobSourcePlatform(str, Enum):
    LINKEDIN = "linkedin"
    JOBSTREET = "jobstreet"
    ONLINEJOBS = "onlinejobs"
    INDEED = "indeed"
    OTHER = "other"


class JobAnalytics(BaseModel):
    """Structured analytics-ready job record with normalized scope-of-work dimensions."""

    # --- Core Identifiers ---
    company_name: str = Field(description="Name of the hiring company")
    nature_of_business: str = Field(
        description="Industry sector. e.g., 'SaaS', 'Healthcare', 'E-commerce', 'FinTech'"
    )

    # --- Geographic & Source Normalization ---
    country_code: Optional[str] = Field(
        None,
        pattern=r"^[A-Z]{2}$",
        description="ISO 3166-1 alpha-2 country code. e.g., 'US', 'GB', 'DE', 'IN'",
    )
    source_platform: Optional[JobSourcePlatform] = Field(
        None, description="Normalized job board/platform where the listing was sourced"
    )

    # --- Timeline Analytics ---
    posted_date: Optional[date] = Field(
        None,
        description="Date the job was first posted. Enables trend analysis, hiring velocity, and stale-job detection.",
    )

    # --- Employment Metadata (Normalized Enums) ---
    employment_type: EmploymentType = Field(
        description="Standardized employment classification"
    )
    work_mode: WorkMode = Field(
        description="Standardized remote/onsite/hybrid classification"
    )

    # --- Compensation (Structured Numeric Range) ---
    salary_min: Optional[int] = Field(
        None,
        ge=0,
        description="Minimum annual/base salary in smallest currency unit or whole units",
    )
    salary_max: Optional[int] = Field(
        None,
        ge=0,
        description="Maximum annual/base salary in smallest currency unit or whole units",
    )
    salary_currency: Optional[str] = Field(
        None,
        pattern=r"^[A-Z]{3}$",
        description="ISO 4217 currency code, e.g., 'USD', 'EUR', 'GBP'",
    )

    # --- Scope of Work Dimensions ---
    technology_stack: Optional[List[str]] = Field(
        None,
        description="Specific technologies, tools, frameworks, APIs, and platforms. "
        "e.g., ['PyTorch', 'LangChain', 'OpenAI API', 'Pinecone', 'CUDA']. "
        "Do NOT include generic terms like 'AI' or 'ML' here.",
    )
    job_domain: Optional[List[str]] = Field(
        None,
        description="Primary engineering domains. e.g., ['Frontend', 'Backend', 'Mobile', "
        "'Cloud/DevOps', 'Data/AI', 'QA/Test', 'Security', 'Embedded/IoT']",
    )
    platform_targets: Optional[List[str]] = Field(
        None,
        description="Target platforms/environments. e.g., ['Web', 'iOS', 'Android', 'AWS', "
        "'Azure', 'GCP', 'Linux Server', 'Windows Desktop']",
    )

    # --- Validators ---
    @field_validator("salary_max")
    @classmethod
    def validate_salary_range(cls, v, info):
        """Ensure salary_max >= salary_min when both are present."""
        if v is not None and info.data.get("salary_min") is not None:
            if v < info.data["salary_min"]:
                raise ValueError(
                    f"salary_max ({v}) must be >= salary_min ({info.data['salary_min']})"
                )
        return v

    @field_validator(
        "technology_stack", "job_domain", "platform_targets", mode="before"
    )
    @classmethod
    def normalize_lists(cls, v):
        """Strip whitespace, deduplicate, and filter empty strings from list fields."""
        if v is None:
            return None
        if isinstance(v, str):
            v = [v]
        cleaned = list(
            dict.fromkeys(item.strip() for item in v if item and item.strip())
        )
        return cleaned or None
