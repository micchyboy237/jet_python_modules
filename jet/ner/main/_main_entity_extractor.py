import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

from jet.ner.entity_extractor import extract_entities_from_text
from pydantic import BaseModel, Field, field_validator
from rich.console import Console

console = Console()

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    class JobEntities(BaseModel):
        # Core fields that will always exist
        job_title: str = Field(description="Official job title/position name")
        key_responsibilities: List[str] = Field(
            description="Main duties and responsibilities of the role"
        )

        # Company Information
        company_name: Optional[str] = Field(
            None, description="Name of the hiring company"
        )
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

        # Compensation
        salary_range: Optional[str] = Field(
            None,
            description="Salary range or compensation information, e.g., '$80K-$120K', 'Competitive'",
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
                # Handle "None specified", "Not mentioned", etc.
                if v.lower() in [
                    "none specified",
                    "not mentioned",
                    "none",
                    "n/a",
                    "not specified",
                ]:
                    return None
                # Convert single string to list
                return [v]
            return v

    # Load first job from JSON
    json_path = Path(
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Apps/my-jobs/saved/jobs.json"
    )
    with open(json_path, "r") as f:
        first_job = json.load(f)[0]

    # Format text with title and details
    text = f"Title: {first_job.get('title', '')}\nOverview:\n{first_job.get('details', '')}"

    # Extract entities from job description
    result = extract_entities_from_text(
        text=text, model_class=JobEntities, temperature=0.0
    )

    # Print results to console
    console.print("\n[bold cyan]Extraction Results:[/bold cyan]")
    console.print(json.dumps(result.model_dump(), indent=2, default=str))

    # ============================================================
    # Save all outputs at the end
    # ============================================================

    # Define file paths
    input_path = OUTPUT_DIR / "input_text.txt"
    schema_path = OUTPUT_DIR / "model_schema.json"
    results_path = OUTPUT_DIR / "extracted_entities.json"
    summary_path = OUTPUT_DIR / "extraction_summary.txt"
    raw_job_path = OUTPUT_DIR / "raw_job_data.json"

    # Save input text
    input_path.write_text(text)
    console.print(
        f"[green]✓[/green] Saved input text: [link=file://{input_path}]{input_path.name}[/link]"
    )

    # Save model schema
    schema = JobEntities.model_json_schema()
    schema_path.write_text(json.dumps(schema, indent=2))
    console.print(
        f"[green]✓[/green] Saved model schema: [link=file://{schema_path}]{schema_path.name}[/link]"
    )

    # Save extracted results as JSON
    results_path.write_text(json.dumps(result.model_dump(), indent=2, default=str))
    console.print(
        f"[green]✓[/green] Saved extracted entities: [link=file://{results_path}]{results_path.name}[/link]"
    )

    # Save results as pretty-printed text summary
    with open(summary_path, "w") as f:
        f.write(f"Entity Extraction Results\n")
        f.write(f"{'=' * 50}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Model: JobEntities\n")
        f.write(f"Temperature: 0.0\n\n")

        data = result.model_dump()
        for key, value in data.items():
            f.write(f"\n{key.replace('_', ' ').title()}:\n")
            f.write(f"{'-' * 30}\n")
            if value is None:
                f.write("Not specified\n")
            elif isinstance(value, list):
                for item in value:
                    f.write(f"  • {item}\n")
            else:
                f.write(f"  {value}\n")

    console.print(
        f"[green]✓[/green] Saved extraction summary: [link=file://{summary_path}]{summary_path.name}[/link]"
    )

    # Save the raw job data for reference
    raw_job_path.write_text(json.dumps(first_job, indent=2, default=str))
    console.print(
        f"[green]✓[/green] Saved raw job data: [link=file://{raw_job_path}]{raw_job_path.name}[/link]"
    )

    console.print(
        f"\n[bold green]All files saved to: [link=file://{OUTPUT_DIR}]{OUTPUT_DIR.name}[/link][/bold green]"
    )


if __name__ == "__main__":
    main()
