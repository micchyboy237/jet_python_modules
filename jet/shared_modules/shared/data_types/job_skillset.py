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
    years_of_experience: Optional[str] = Field(
        None,
        description="Required experience duration. e.g., '3+ years', '5-7 years', 'Entry Level'",
    )
    # Consolidated Skills List
    skills: Optional[List[SkillItem]] = Field(
        None,
        description="List of all skills with type and priority. Example: [{'name': 'Python', 'type': 'hard', 'priority': 'required'}, {'name': 'Communication', 'type': 'soft', 'priority': 'required'}]",
    )
