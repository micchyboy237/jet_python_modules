# jet/shared_modules/shared/data_types/job_analytics.py
from typing import List, Optional, Union

from pydantic import BaseModel, Field


class JobAnalytics(BaseModel):
    # Core fields that will always exist
    job_title: str = Field(description="Official job title/position name")
    key_responsibilities: List[str] = Field(
        description="Main duties and responsibilities of the role"
    )

    # Company Information
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

    # Job Overview
    department: Optional[str] = Field(
        None, description="Department or team within the company"
    )
    job_description_summary: Optional[str] = Field(
        None, description="2-3 sentence overview of the role and its purpose"
    )

    # Location & Work Arrangement
    job_location: Optional[str] = Field(
        None, description="Primary work location - city, state/country"
    )
    remote_work_policy: Optional[str] = Field(
        None,
        description="e.g., 'Fully Remote', 'Hybrid (2 days/week in office)', 'On-site'",
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
    salary_frequency: Optional[str] = Field(
        None,
        description="Pay frequency. MUST be one of: 'hourly', 'daily', 'weekly', 'biweekly', 'semi_monthly', 'monthly', 'annual', 'other'.",
    )

    # Classification
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

    # Technical Requirements (lists that should be lists)
    required_skills: Optional[List[str]] = Field(
        None, description="Must-have technical and soft skills"
    )
    preferred_skills: Optional[List[str]] = Field(
        None, description="Nice-to-have but not required skills"
    )
    technology_stack: Optional[List[str]] = Field(
        None, description="Specific technologies, tools, frameworks mentioned"
    )

    # These fields might come back as strings or lists from the LLM
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

    # Benefits
    employee_benefits: Optional[Union[str, List[str]]] = Field(
        None, description="Benefits, perks, and compensation extras offered"
    )

    # Application Process
    application_instructions: Optional[str] = Field(
        None,
        description="How to apply, including any links, emails, or special instructions",
    )
    application_deadline: Optional[str] = Field(
        None, description="Application deadline if specified"
    )
