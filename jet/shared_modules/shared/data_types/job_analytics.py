from datetime import date, datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class EmploymentType(str, Enum):
    FULL_TIME = "full_time"
    PART_TIME = "part_time"
    CONTRACT = "contract"
    INTERNSHIP = "internship"
    OTHER = "other"


class WorkMode(str, Enum):
    REMOTE = "remote"
    ONSITE = "onsite"
    HYBRID = "hybrid"
    OTHER = "other"


class JobSourcePlatform(str, Enum):
    LINKEDIN = "linkedin"
    JOBSTREET = "jobstreet"
    ONLINEJOBS = "onlinejobs"
    INDEED = "indeed"
    OTHER = "other"


class SalaryFrequency(str, Enum):
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    BIWEEKLY = "biweekly"
    SEMI_MONTHLY = "semi_monthly"
    MONTHLY = "monthly"
    ANNUAL = "annual"
    OTHER = "other"


class SeniorityLevel(str, Enum):
    INTERN = "intern"
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"
    LEAD = "lead"
    PRINCIPAL = "principal"
    EXECUTIVE = "executive"
    OTHER = "other"


class JobAnalytics(BaseModel):
    """Structured analytics-ready job record with normalized scope-of-work dimensions."""

    model_config = ConfigDict(use_enum_values=True, populate_by_name=True)

    # --- Company & Location ---
    company_name: Optional[str] = Field(
        None,
        description="Name of the hiring company. e.g., 'TechNova Solutions', 'Google'",
    )
    nature_of_business: Optional[str] = Field(
        None,
        description="Industry sector. e.g., 'SaaS', 'HealthTech', 'E-commerce', 'FinTech'",
    )
    country_code: Optional[str] = Field(
        None,
        pattern=r"^[A-Z]{2}$",
        description="ISO 3166-1 alpha-2 country code. e.g., 'US', 'GB', 'DE', 'IN'",
    )
    source_platform: Optional[JobSourcePlatform] = Field(
        None,
        description="Normalized job board. MUST be one of: 'linkedin', 'jobstreet', 'onlinejobs', 'indeed', 'other'.",
    )

    # --- Dates & Type ---
    posted_date: Optional[date] = Field(
        None,
        description="Date the job was first posted in ISO 8601 format (YYYY-MM-DD). e.g., '2024-05-15'",
    )
    employment_type: Optional[EmploymentType] = Field(
        None,
        description="Standardized employment type. MUST be one of: 'full_time', 'part_time', 'contract', 'internship', 'other'.",
    )
    work_mode: Optional[WorkMode] = Field(
        None,
        description="Standardized work mode. MUST be one of: 'remote', 'onsite', 'hybrid', 'other'.",
    )

    # --- Compensation ---
    salary_min: Optional[int] = Field(
        None,
        ge=0,
        description="Minimum annual/base salary as an integer. e.g., 120000",
    )
    salary_max: Optional[int] = Field(
        None,
        ge=0,
        description="Maximum annual/base salary as an integer. e.g., 160000",
    )
    salary_currency: Optional[str] = Field(
        None,
        pattern=r"^[A-Z]{3}$",
        description="ISO 4217 currency code. e.g., 'USD', 'EUR', 'GBP', 'PHP'",
    )
    salary_frequency: Optional[SalaryFrequency] = Field(
        None,
        description="Pay frequency. MUST be one of: 'hourly', 'daily', 'weekly', 'biweekly', 'semi_monthly', 'monthly', 'annual', 'other'.",
    )

    # --- Experience & Seniority ---
    seniority_level: Optional[SeniorityLevel] = Field(
        None,
        description="Career level required. MUST be one of: 'intern', 'junior', 'mid', 'senior', 'lead', 'principal', 'executive', 'other'. Note: 'Entry Level' maps to 'junior'.",
    )
    years_experience_min: Optional[float] = Field(
        None,
        ge=0,
        description="Minimum years of relevant experience required. Can be fractional. e.g., 1.5",
    )
    years_experience_max: Optional[float] = Field(
        None,
        ge=0,
        description="Maximum years of experience expected (if capped). Can be fractional. e.g., 7.5",
    )
    experience_description: Optional[str] = Field(
        None,
        description="Raw text description of experience requirements if specific years are not mentioned. e.g., 'Proven track record in...'",
    )

    # --- Technology Stack (Categorized) ---
    required_technologies: Optional[List[str]] = Field(
        None,
        description="Mandatory technologies, tools, frameworks, and libraries. The candidate MUST have these. e.g., ['Python', 'AWS', 'React']",
    )
    preferred_technologies: Optional[List[str]] = Field(
        None,
        description="Nice-to-have technologies. Candidates with these are preferred but not required. e.g., ['Docker', 'Kubernetes']",
    )
    optional_technologies: Optional[List[str]] = Field(
        None,
        description="Other technologies mentioned in the stack but not explicitly categorized as required or preferred.",
    )

    # --- Domain & Platform ---
    job_domain: Optional[List[str]] = Field(
        None,
        description="Primary engineering domains. e.g., ['Backend', 'Frontend', 'Data/AI', 'DevOps', 'Mobile']",
    )
    platform_targets: Optional[List[str]] = Field(
        None,
        description="Target deployment platforms or environments. e.g., ['Web', 'iOS', 'Android', 'AWS', 'Linux']",
    )

    # --- Validators ---

    @field_validator(
        "source_platform",
        "employment_type",
        "work_mode",
        "seniority_level",
        mode="before",
    )
    @classmethod
    def normalize_enum_strings(cls, v):
        """Normalize common LLM output variations to match Enum values."""
        if v is None:
            return None
        if isinstance(v, str):
            cleaned = v.lower().replace("-", "_").replace(" ", "_")
            mappings = {
                "full_time": "full_time",
                "fulltime": "full_time",
                "part_time": "part_time",
                "parttime": "part_time",
                "remote": "remote",
                "on_site": "onsite",
                "on-site": "onsite",
                "onsite": "onsite",
                "hybrid": "hybrid",
                "linkedin": "linkedin",
                "job_street": "jobstreet",
                "jobstreet": "jobstreet",
                "online_jobs": "onlinejobs",
                "onlinejobs": "onlinejobs",
                "indeed": "indeed",
                "intern": "intern",
                "internship": "intern",
                # Map Entry/Graduate to Junior to avoid redundancy
                "entry": "junior",
                "entry_level": "junior",
                "graduate": "junior",
                "new_grad": "junior",
                "associate": "junior",
                "junior": "junior",
                "jr": "junior",
                "jr_": "junior",
                "mid": "mid",
                "mid_level": "mid",
                "mid-level": "mid",
                "senior": "senior",
                "sr": "senior",
                "sr_": "senior",
                "lead": "lead",
                "team_lead": "lead",
                "principal": "principal",
                "staff": "principal",
                "executive": "executive",
                "c_level": "executive",
                "cto": "executive",
                "ceo": "executive",
            }
            return mappings.get(cleaned, cleaned)
        return v

    @field_validator("posted_date", mode="before")
    @classmethod
    def coerce_posted_date(cls, v):
        if v is None:
            return None
        if isinstance(v, date) and not isinstance(v, datetime):
            return v
        if isinstance(v, datetime):
            return v.date()
        if isinstance(v, str):
            v = v.strip()
            if not v:
                return None
            try:
                return datetime.fromisoformat(v.replace("Z", "+00:00")).date()
            except ValueError:
                pass
            try:
                return date.fromisoformat(v)
            except ValueError:
                raise ValueError(f"Cannot parse posted_date: {v!r}")
        raise ValueError(f"Unsupported posted_date type: {type(v)}")

    @field_validator("salary_max")
    @classmethod
    def validate_salary_range(cls, v, info):
        if v is not None and info.data.get("salary_min") is not None:
            if v < info.data["salary_min"]:
                raise ValueError(
                    f"salary_max ({v}) must be >= salary_min ({info.data['salary_min']})"
                )
        return v

    @field_validator("years_experience_max")
    @classmethod
    def validate_experience_range(cls, v, info):
        if v is not None and info.data.get("years_experience_min") is not None:
            if v < info.data["years_experience_min"]:
                raise ValueError(
                    f"years_experience_max ({v}) must be >= years_experience_min ({info.data['years_experience_min']})"
                )
        return v

    @field_validator(
        "required_technologies",
        "preferred_technologies",
        "optional_technologies",
        "job_domain",
        "platform_targets",
        mode="before",
    )
    @classmethod
    def normalize_lists(cls, v):
        if v is None:
            return None
        if isinstance(v, str):
            v = [v]
        cleaned = list(
            dict.fromkeys(item.strip() for item in v if item and item.strip())
        )
        return cleaned or None
