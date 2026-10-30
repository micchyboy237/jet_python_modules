# jet/shared_modules/shared/data_types/job_analytics.py
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field


class SkillItem(BaseModel):
    """A single skill with its type and priority level."""

    name: str = Field(
        description="The name of the skill. e.g., 'Python', 'Communication', 'AWS'"
    )
    type: Literal["hard", "soft"] = Field(
        description="Whether the skill is technical (hard) or behavioral (soft)."
    )
    priority: Literal["required", "preferred"] = Field(
        description="Whether the skill is mandatory (required) or nice-to-have (preferred)."
    )


class JobAnalytics(BaseModel):
    # --- Core Identification ---
    # job_id: Optional[str] = Field(
    #     None, description="Unique identifier for the job posting if available."
    # )
    # job_title: str = Field(
    #     description="Official job title/position name. e.g., 'Machine Learning Engineer Associate'"
    # )
    # posting_date: Optional[str] = Field(
    #     None, description="Date the job was posted. e.g., '2024-10-30'"
    # )
    # application_url: Optional[str] = Field(
    #     None, description="Direct URL to apply for the job."
    # )

    # --- Company Information ---
    company_name: Optional[str] = Field(
        None, description="Name of the hiring company. e.g., 'TechCorp Inc.'"
    )
    company_summary: Optional[str] = Field(
        None, description="Brief overview of what the company does."
    )
    nature_of_business: Optional[str] = Field(
        None,
        description="Industry sector. e.g., 'SaaS', 'Healthcare', 'FinTech', 'E-commerce'",
    )
    company_size: Optional[str] = Field(
        None,
        description="Approximate company size. e.g., 'Startup', '50-200 employees', 'Enterprise'",
    )

    # --- Job Overview ---
    department: Optional[str] = Field(
        None, description="Department or team. e.g., 'AI Research', 'Engineering'"
    )
    job_description_summary: Optional[str] = Field(
        None, description="2-3 sentence overview of the role and its purpose."
    )
    key_responsibilities: List[str] = Field(
        default_factory=list,
        description="Main duties and responsibilities. e.g., ['Develop ML models', 'Collaborate with data scientists']",
    )

    # --- Location & Work Arrangement ---
    job_location: Optional[str] = Field(
        None, description="Primary work location. e.g., 'San Francisco, CA', 'Remote'"
    )
    remote_work_policy: Optional[str] = Field(
        None,
        description="Work arrangement. e.g., 'Fully Remote', 'Hybrid (2 days/week)', 'On-site'",
    )

    # --- Compensation ---
    salary_min: Optional[int] = Field(
        None, ge=0, description="Minimum annual/base salary. e.g., 120000"
    )
    salary_max: Optional[int] = Field(
        None, ge=0, description="Maximum annual/base salary. e.g., 160000"
    )
    salary_currency: Optional[str] = Field(
        None, pattern=r"^[A-Z]{3}$", description="ISO currency code. e.g., 'USD', 'EUR'"
    )
    salary_frequency: Optional[str] = Field(
        None,
        description="Pay frequency. e.g., 'annual', 'hourly', 'monthly'",
    )

    # --- Classification ---
    employment_type: Optional[str] = Field(
        None, description="e.g., 'Full-time', 'Part-time', 'Contract', 'Internship'"
    )
    experience_level: Optional[str] = Field(
        None, description="e.g., 'Entry Level', 'Mid-Level', 'Senior', 'Lead'"
    )
    schedule_type: Optional[str] = Field(
        None, description="e.g., 'Flexible hours', '9-to-5', 'Shift work'"
    )

    # --- Requirements (Refactored to reduce redundancy) ---

    # Consolidated Skills List
    skills: Optional[List[SkillItem]] = Field(
        None,
        description="List of all skills with type and priority. Example: [{'name': 'Python', 'type': 'hard', 'priority': 'required'}, {'name': 'Communication', 'type': 'soft', 'priority': 'required'}]",
    )

    # Distinct Qualification Fields
    years_of_experience: Optional[str] = Field(
        None,
        description="Required experience duration. e.g., '3+ years', '5-7 years', 'Entry Level'",
    )
    education_requirements: Optional[Union[str, List[str]]] = Field(
        None,
        description="Required education. e.g., 'Bachelor's in Computer Science', 'Master's degree preferred'",
    )
    certifications_required: Optional[Union[str, List[str]]] = Field(
        None,
        description="Required certifications. e.g., 'AWS Certified Machine Learning', 'PMP'",
    )
    language_requirements: Optional[Union[str, List[str]]] = Field(
        None,
        description="Required spoken/written languages. e.g., 'English (Fluent)', 'Spanish (Conversational)'",
    )

    # --- Benefits & Application ---
    employee_benefits: Optional[Union[str, List[str]]] = Field(
        None,
        description="Perks and benefits. e.g., ['Health Insurance', '401k Matching', 'Remote Work Stipend']",
    )
    application_instructions: Optional[str] = Field(
        None,
        description="How to apply. e.g., 'Apply via LinkedIn', 'Send resume to jobs@company.com'",
    )
    application_deadline: Optional[str] = Field(
        None, description="Application deadline. e.g., '2024-11-30'"
    )
