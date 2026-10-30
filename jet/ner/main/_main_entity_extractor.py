import json
import shutil
from datetime import datetime
from pathlib import Path

from jet.ner.entity_extractor import extract_entities_from_text
from rich.console import Console
from shared.data_types.job_skillset import JobSkillset

console = Console()

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    # Load first job from JSON where "entities" is null, if all have entities, use first job
    json_path = Path(
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Apps/my-jobs/generated/run_load_jobs_list/jobs.json"
    )
    with open(json_path, "r") as f:
        jobs = json.load(f)["jobs"]
        first_job = next((job for job in jobs if job.get("entities") is None), None)
        if first_job is None:
            first_job = jobs[0]

    # Format text with title and details
    text = f"Title: {first_job.get('title', '')}\nOverview:\n{first_job.get('details', '')}"

    # Extract entities from job description
    result = extract_entities_from_text(
        text=text, model_class=JobSkillset, temperature=0.3
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
    schema = JobSkillset.model_json_schema()
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
        f.write(f"Model: JobSkillset\n")
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
