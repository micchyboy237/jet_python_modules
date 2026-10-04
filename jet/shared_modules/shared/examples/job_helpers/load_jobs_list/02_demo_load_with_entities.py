"""
Definition Summary:
    Demonstrates loading jobs with their extracted entities included.
    Entities contain structured data like work_mode, seniority_level, required_technologies, etc.
    Uses LEFT JOIN to combine job metadata with entity data from job_entities table.

Usage Examples:
    # Load jobs with entities
    >>> jobs = load_jobs_list(db_client=db_client, include_entities=True)

    # Access entity data
    >>> for job in jobs:
    ...     if job.get("entities"):
    ...         print(job["entities"].get("work_mode"))

Span Hierarchies:
    📦 load-jobs-with-entities (CHAIN)
    │
    ├── 🔍 SQL Query Execution
    │   ├── SELECT m.*, e.entities FROM public.jobs m
    │   └── LEFT JOIN public.job_entities e ON m.id = e.id
    │
    └── 🔄 Entity Merging
        └── Merges entity JSON into each JobData dict
"""

import shutil
from pathlib import Path

from jet.db.postgres.pgvector import PgVectorClient
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree
from shared.job_helpers import load_jobs_list

console = Console()

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    console.print(
        Panel.fit("🔍 Demo 02: Loading Jobs with Entities", style="bold cyan")
    )

    # Initialize database client
    db_client = PgVectorClient(dbname="jobs_db3")

    console.print("\n[yellow]Loading jobs WITH extracted entities...[/yellow]")

    # Load jobs with entities
    jobs = load_jobs_list(db_client=db_client, include_entities=True)

    console.print(f"\n[green]✓ Loaded {len(jobs)} jobs with entities[/green]")

    # Display jobs with entity highlights
    if jobs:
        table = Table(
            title="Jobs with Entity Highlights (First 5)",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("ID", style="cyan")
        table.add_column("Title", style="green")
        table.add_column("Work Mode", style="yellow")
        table.add_column("Seniority", style="blue")
        table.add_column("Technologies", style="magenta")

        for job in jobs[:5]:
            entities = job.get("entities") or {}

            work_mode = entities.get("work_mode", "N/A")
            seniority = entities.get("seniority_level", "N/A")
            techs = entities.get("required_technologies", [])
            tech_str = ", ".join(techs[:3]) if techs else "N/A"

            table.add_row(
                str(job.get("id", "N/A")),
                str(job.get("title", "N/A"))[:40],
                str(work_mode),
                str(seniority),
                tech_str[:40],
            )

        console.print(table)

        # Show detailed entity structure for first job
        first_job_with_entities = next((j for j in jobs if j.get("entities")), None)
        if first_job_with_entities:
            console.print("\n[bold]Sample Entity Structure:[/bold]")
            tree = Tree(f"Job: {first_job_with_entities.get('title', 'Unknown')}")
            entities = first_job_with_entities["entities"]

            for key, value in entities.items():
                if isinstance(value, list):
                    branch = tree.add(f"[cyan]{key}[/cyan]: {len(value)} items")
                    for item in value[:3]:
                        branch.add(str(item))
                else:
                    tree.add(f"[cyan]{key}[/cyan]: [yellow]{value}[/yellow]")

            console.print(tree)

        # Save sample output
        output_file = OUTPUT_DIR / "jobs_with_entities_sample.json"
        import json

        with open(output_file, "w") as f:
            json.dump([dict(job) for job in jobs[:5]], f, indent=2, default=str)
        console.print(
            f"\n[spring_green1]✓ Sample data saved to: {output_file}[/spring_green1]"
        )

    # Statistics
    jobs_with_entities = sum(1 for j in jobs if j.get("entities"))
    console.print(f"\n[bold]Entity Statistics:[/bold]")
    console.print(f"  • Total jobs: {len(jobs)}")
    console.print(f"  • Jobs with entities: {jobs_with_entities}")
    console.print(
        f"  • Coverage: {(jobs_with_entities / len(jobs) * 100) if jobs else 0:.1f}%"
    )


if __name__ == "__main__":
    main()
