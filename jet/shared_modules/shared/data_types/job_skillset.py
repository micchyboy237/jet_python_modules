# job_skillset.py

from typing import List, Optional

from pydantic import BaseModel, Field


class JobSkillset(BaseModel):
    years_of_experience: Optional[str] = Field(
        None, description="Experience duration. e.g., '3+ years', 'Entry Level'."
    )

    # Hard Skills: Specific Tools & Technologies (Proper Nouns Only)
    tools_required: Optional[List[str]] = Field(
        None,
        description=(
            "Mandatory specific software, languages, or platforms. "
            "STRICT: Entity names ONLY. No verbs, no proficiency levels. "
            "e.g., ['Python', 'AWS', 'Postman'], NOT ['Proficient in Python']."
        ),
    )

    tools_preferred: Optional[List[str]] = Field(
        None,
        description=(
            "Nice-to-have specific software, languages, or platforms. "
            "STRICT: Entity names ONLY. e.g., ['Azure', 'Docker']."
        ),
    )

    # Hard Skills: Methodologies, Domains & Practices
    methods_required: Optional[List[str]] = Field(
        None,
        description=(
            "Mandatory technical processes, domains, or practices. "
            "e.g., ['Agile', 'Manual Testing', 'Machine Learning', 'API Development']."
        ),
    )

    methods_preferred: Optional[List[str]] = Field(
        None,
        description=(
            "Nice-to-have technical processes, domains, or practices. "
            "e.g., ['DevOps', 'CI/CD', 'Cloud Architecture']."
        ),
    )

    # Soft Skills: Behavioral & Interpersonal Traits
    soft_required: Optional[List[str]] = Field(
        None,
        description=(
            "Mandatory behavioral traits. Can be concise phrases. "
            "e.g., ['Communication', 'Problem-solving', 'Teamwork']."
        ),
    )

    soft_preferred: Optional[List[str]] = Field(
        None,
        description=(
            "Nice-to-have behavioral traits. e.g., ['Leadership', 'Mentoring']."
        ),
    )
