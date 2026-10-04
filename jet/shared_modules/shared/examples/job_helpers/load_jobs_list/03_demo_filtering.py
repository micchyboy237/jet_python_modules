"""
Definition Summary:
    Demonstrates filtering jobs using where_conditions parameter.
    Supports exact matches, NOT_NULL checks, and IS_NULL checks.
    All filtering happens at the database level for efficiency.

Usage Examples:
    # Filter by exact match
    >>> jobs = load_jobs_list(db_client=db_client,
    ...                       where_conditions={"job_type": "Full Time"})

    # Filter for non-null values
    >>> jobs = load_jobs_list(db_client=db_client,
    ...                       where_conditions={"salary": "NOT_NULL"})

    # Filter for null values
    >>> jobs = load_jobs_list(db_client=db_client,
    ...                       where_conditions={"hours_per_week": "IS_NULL"})

    # Combine multiple filters
    >>> jobs = load_jobs_list(db_client=db_client,
    ...                       where_conditions={
    ...                           "job_type": "Contract",
    ...                           "domain": "linkedin.com"
    ...                       })

Span Hierarchies:
    📦 load-jobs-filtered (CHAIN)
    │
    └── 🔍 SQL Query Execution
        ├── WHERE job_type = %s AND domain = %s
        ├── WHERE salary IS NOT NULL AND salary != ''
        └── WHERE hours_per_week IS NULL OR hours_per_week = ''
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
    console.print(
        Panel.fit("🔍 Demo 03: Filtering Jobs with where_conditions", style="bold cyan")
    )

    # Initialize database client
    db_client = PgVectorClient(dbname="jobs_db3")

    # Example 1: Filter by job type
    console.print(
        "\n[bold yellow]Example 1: Filter by Job Type = 'Contract'[/bold yellow]"
    )
    contract_jobs = load_jobs_list(
        db_client=db_client, where_conditions={"job_type": "Contract"}
    )
    console.print(f"[green]✓ Found {len(contract_jobs)} contract jobs[/green]")

    if contract_jobs:
        table1 = Table(
            title="Contract Jobs (First 5)",
            show_header=True,
            header_style="bold magenta",
        )
        table1.add_column("ID", style="cyan")
        table1.add_column("Title", style="green")
        table1.add_column("Company", style="yellow")

        for job in contract_jobs[:5]:
            table1.add_row(
                str(job.get("id", "N/A")),
                str(job.get("title", "N/A"))[:40],
                str(job.get("company", "N/A"))[:30],
            )
        console.print(table1)

    # Example 2: Filter for jobs with salary information
    console.print(
        "\n[bold yellow]Example 2: Filter for Jobs with Salary (NOT_NULL)[/bold yellow]"
    )
    jobs_with_salary = load_jobs_list(
        db_client=db_client, where_conditions={"salary": "NOT_NULL"}
    )
    console.print(
        f"[green]✓ Found {len(jobs_with_salary)} jobs with salary info[/green]"
    )

    if jobs_with_salary:
        table2 = Table(
            title="Jobs with Salary (First 5)",
            show_header=True,
            header_style="bold magenta",
        )
        table2.add_column("ID", style="cyan")
        table2.add_column("Title", style="green")
        table2.add_column("Salary", style="yellow")
        table2.add_column("Job Type", style="blue")

        for job in jobs_with_salary[:5]:
            table2.add_row(
                str(job.get("id", "N/A")),
                str(job.get("title", "N/A"))[:40],
                str(job.get("salary", "N/A")),
                str(job.get("job_type", "N/A")),
            )
        console.print(table2)

    # Example 3: Filter for jobs without hours_per_week
    console.print(
        "\n[bold yellow]Example 3: Filter for Jobs Missing Hours (IS_NULL)[/bold yellow]"
    )
    jobs_no_hours = load_jobs_list(
        db_client=db_client, where_conditions={"hours_per_week": "IS_NULL"}
    )
    console.print(
        f"[green]✓ Found {len(jobs_no_hours)} jobs without hours info[/green]"
    )

    # Example 4: Combined filters
    console.print(
        "\n[bold yellow]Example 4: Combined Filters (LinkedIn + Full Time)[/bold yellow]"
    )
    combined_jobs = load_jobs_list(
        db_client=db_client,
        where_conditions={"domain": "linkedin.com", "job_type": "Full Time"},
    )
    console.print(
        f"[green]✓ Found {len(combined_jobs)} LinkedIn full-time jobs[/green]"
    )

    # Save all results
    import json

    output_file = OUTPUT_DIR / "filtered_jobs_results.json"
    results = {
        "contract_jobs_count": len(contract_jobs),
        "jobs_with_salary_count": len(jobs_with_salary),
        "jobs_no_hours_count": len(jobs_no_hours),
        "combined_filter_count": len(combined_jobs),
        "sample_contract_jobs": [dict(j) for j in contract_jobs[:3]],
        "sample_salary_jobs": [dict(j) for j in jobs_with_salary[:3]],
    }
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    console.print(f"\n[spring_green1]✓ Results saved to: {output_file}[/spring_green1]")

    # Summary
    console.print("\n[bold]Filter Summary:[/bold]")
    console.print(f"  • Contract jobs: {len(contract_jobs)}")
    console.print(f"  • Jobs with salary: {len(jobs_with_salary)}")
    console.print(f"  • Jobs missing hours: {len(jobs_no_hours)}")
    console.print(f"  • LinkedIn full-time: {len(combined_jobs)}")


if __name__ == "__main__":
    main()
