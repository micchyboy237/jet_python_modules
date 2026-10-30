# job_skillset.py

from typing import List, Optional

from pydantic import BaseModel, Field


class JobSkillset(BaseModel):
    years_of_experience: Optional[str] = Field(
        None,
        description=(
            "Required experience duration. Extract exact phrasing if possible. "
            "Examples: '3+ years', '5-7 years', 'Entry Level', 'Junior (0-2 years)', "
            "'Senior (8+ years)'"
        ),
    )

    hard_required_skills: Optional[List[str]] = Field(
        None,
        description=(
            "Mandatory technical skills, tools, languages, or platforms. "
            "These are concrete, measurable abilities essential for the job. "
            "Examples: 'Python', 'SQL', 'AWS', 'Docker', 'Machine Learning', 'AI', "
            "'TensorFlow', 'PyTorch', 'NLP', 'Computer Vision', 'Deep Learning', 'LLMs', "
            "'React', 'Java', 'Kubernetes', 'Tableau', 'Salesforce'."
        ),
    )

    hard_preferred_skills: Optional[List[str]] = Field(
        None,
        description=(
            "Nice-to-have technical skills, tools, languages, or platforms. "
            "These strengthen the candidate's profile but are not deal-breakers. "
            "Examples: 'Azure', 'GCP', 'NoSQL', 'GraphQL', 'CI/CD pipelines', "
            "'Jest', 'Cypress', 'Basic HTML/CSS', 'Figma'."
        ),
    )

    soft_required_skills: Optional[List[str]] = Field(
        None,
        description=(
            "Mandatory behavioral, interpersonal, or cognitive attributes. "
            "In job postings, these often appear as phrases describing how one works. "
            "Examples: 'Communication', 'Teamwork', 'Leadership', 'Problem-solving', "
            "'Adaptability', 'Time management', 'Critical thinking', 'Emotional intelligence', "
            "'Collaboration', 'Mentoring', 'Ability to work independently', "
            "'Cross-functional collaboration', 'Strong attention to detail', "
            "'Excellent verbal and written communication', 'Ability to prioritize multiple tasks'."
        ),
    )

    soft_preferred_skills: Optional[List[str]] = Field(
        None,
        description=(
            "Nice-to-have behavioral, interpersonal, or cognitive attributes. "
            "These are desirable traits that would make a candidate stand out. "
            "Examples: 'Public speaking', 'Conflict resolution', 'Strategic thinking', "
            "'Self-starter', 'Ability to mentor junior staff', 'Experience working in agile teams', "
            "'Creativity', 'Negotiation skills'."
        ),
    )
