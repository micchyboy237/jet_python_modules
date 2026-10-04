"""
Definition Summary:
    Demonstrates basic usage of load_jobs_list() to retrieve all jobs from the database.
    This is the simplest form - no filters, no entities, just raw job metadata.

Usage Examples:
    # Load all jobs (default behavior)
    >>> jobs = load_jobs_list(db_client=db_client)

    # Load jobs with specific table name
    >>> jobs = load_jobs_list(db_client=db_client, table_name="jobs")

Span Hierarchies:
    📦 load-jobs-basic (CHAIN)
    │
    └── 🔍 SQL Query Execution
        ├── SELECT * FROM public.jobs
        └── Returns: list[JobData]
"""

import shutil
from pathlib import Path

from jet.db.postgres.pgvector import PgVectorClient
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from shared.job_helpers import load_jobs_list

console = Console()

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    console.print(Panel.fit("🔍 Demo 01: Basic Job List Loading", style="bold cyan"))

    # Initialize database client
    db_client = PgVectorClient(dbname="jobs_db3")

    console.print("\n[yellow]Loading all jobs from database...[/yellow]")

    # Load all jobs without filters
    jobs = load_jobs_list(db_client=db_client)

    console.print(f"\n[green]✓ Loaded {len(jobs)} jobs[/green]")

    # Display summary table
    if jobs:
        table = Table(
            title="Sample Jobs (First 5)", show_header=True, header_style="bold magenta"
        )
        table.add_column("ID", style="cyan")
        table.add_column("Title", style="green")
        table.add_column("Company", style="yellow")
        table.add_column("Job Type", style="blue")
        table.add_column("Posted Date", style="magenta")

        for job in jobs[:5]:
            table.add_row(
                str(job.get("id", "N/A")),
                str(job.get("title", "N/A"))[:40],
                str(job.get("company", "N/A"))[:30],
                str(job.get("job_type", "N/A")),
                str(job.get("posted_date", "N/A"))[:10]
                if job.get("posted_date")
                else "N/A",
            )

        console.print(table)

        # Save sample output
        output_file = OUTPUT_DIR / "basic_jobs_sample.json"
        import json

        with open(output_file, "w") as f:
            json.dump([dict(job) for job in jobs[:10]], f, indent=2, default=str)
        console.print(
            f"\n[spring_green1]✓ Sample data saved to: {output_file}[/spring_green1]"
        )

    # Show statistics
    console.print("\n[bold]Database Statistics:[/bold]")
    console.print(f"  • Total jobs loaded: {len(jobs)}")

    if jobs:
        companies = set(job.get("company") for job in jobs if job.get("company"))
        job_types = set(job.get("job_type") for job in jobs if job.get("job_type"))
        console.print(f"  • Unique companies: {len(companies)}")
        console.print(f"  • Job types found: {len(job_types)}")


if __name__ == "__main__":
    main()
