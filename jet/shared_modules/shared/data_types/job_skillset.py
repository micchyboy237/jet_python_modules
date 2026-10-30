from typing import List, Literal, Optional

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


class JobSkillset(BaseModel):
    min_years_experience: Optional[int] = Field(
        None,
        description="Minimum required years of experience. e.g., 3, 5, 0 for entry level",
        ge=0,  # Ensures non-negative values
    )
    max_years_experience: Optional[int] = Field(
        None,
        description="Maximum years of experience (if applicable). Leave None if no upper limit.",
        ge=0,  # Ensures non-negative values
    )
    # Consolidated Skills List
    skills: Optional[List[SkillItem]] = Field(
        None,
        description="List of all skills with type and priority. Example: [{'name': 'Python', 'type': 'hard', 'priority': 'required'}, {'name': 'Communication', 'type': 'soft', 'priority': 'required'}]",
    )
