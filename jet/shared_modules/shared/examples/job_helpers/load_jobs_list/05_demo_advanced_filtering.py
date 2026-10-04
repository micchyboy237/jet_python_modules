"""
Definition Summary:
    Demonstrates advanced filtering by combining multiple parameters:
    - where_conditions (exact matches, NOT_NULL, IS_NULL)
    - posted_after / posted_before (date ranges)
    - include_entities (join with job_entities table)

    Shows how to build complex queries for specific use cases like finding
    recent remote jobs with salary information.

Usage Examples:
    # Recent remote jobs with salary info
    >>> from datetime import datetime, timedelta
    >>> recent_remote_paid = load_jobs_list(
    ...     db_client=db_client,
    ...     posted_after=datetime.now() - timedelta(days=14),
    ...     where_conditions={
    ...         "salary": "NOT_NULL",
    ...         "job_type": "Full Time"
    ...     },
    ...     include_entities=True
    ... )

    # Jobs missing critical info
    >>> incomplete_jobs = load_jobs_list(
    ...     db_client=db_client,
    ...     where_conditions={
    ...         "salary": "IS_NULL",
    ...         "hours_per_week": "IS_NULL"
    ...     }
    ... )

Span Hierarchies:
    📦 load-jobs-advanced-filtered (CHAIN)
    │
    ├── 🔍 SQL Query Execution
    │   ├── WHERE posted_date >= %s
    │   ├── AND salary IS NOT NULL AND salary != ''
    │   └── AND job_type = %s
    │
    └── 🔄 Optional Entity Join
        └── LEFT JOIN public.job_entities ON jobs.id = job_entities.id
"""

import shutil
from datetime import datetime, timedelta
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
        Panel.fit("🔍 Demo 05: Advanced Filtering Combinations", style="bold cyan")
    )

    # Initialize database client
    db_client = PgVectorClient(dbname="jobs_db3")

    # Example 1: Recent full-time jobs with salary
    console.print(
        "\n[bold yellow]Example 1: Recent Full-Time Jobs with Salary (Last 14 Days)[/bold yellow]"
    )
    fourteen_days_ago = datetime.now() - timedelta(days=14)
    recent_ft_salary = load_jobs_list(
        db_client=db_client,
        posted_after=fourteen_days_ago,
        where_conditions={"salary": "NOT_NULL", "job_type": "Full Time"},
    )
    console.print(f"[green]✓ Found {len(recent_ft_salary)} jobs[/green]")

    if recent_ft_salary:
        table1 = Table(
            title="Recent Full-Time Jobs with Salary",
            show_header=True,
            header_style="bold magenta",
        )
        table1.add_column("ID", style="cyan")
        table1.add_column("Title", style="green")
        table1.add_column("Company", style="yellow")
        table1.add_column("Salary", style="blue")
        table1.add_column("Posted", style="magenta")

        for job in recent_ft_salary[:5]:
            posted = str(job.get("posted_date", ""))[:10]
            table1.add_row(
                str(job.get("id", "N/A")),
                str(job.get("title", "N/A"))[:35],
                str(job.get("company", "N/A"))[:25],
                str(job.get("salary", "N/A")),
                posted,
            )
        console.print(table1)

    # Example 2: Remote jobs with entities
    console.print(
        "\n[bold yellow]Example 2: Remote Jobs with Entity Data[/bold yellow]"
    )
    remote_jobs = load_jobs_list(
        db_client=db_client,
        where_conditions={"domain": "linkedin.com"},
        include_entities=True,
    )

    # Filter for remote work mode from entities
    remote_with_entities = [
        j
        for j in remote_jobs
        if j.get("entities") and j["entities"].get("work_mode") == "remote"
    ]
    console.print(
        f"[green]✓ Found {len(remote_with_entities)} remote LinkedIn jobs[/green]"
    )

    if remote_with_entities:
        console.print("\n[bold]Sample Remote Job Details:[/bold]")
        sample_job = remote_with_entities[0]
        tree = Tree(f"[cyan]{sample_job.get('title', 'Unknown')}[/cyan]")
        tree.add(f"[yellow]Company:[/yellow] {sample_job.get('company', 'N/A')}")
        tree.add(f"[yellow]Salary:[/yellow] {sample_job.get('salary', 'N/A')}")

        entities = sample_job.get("entities", {})
        if entities:
            ent_branch = tree.add("[green]Entities[/green]")
            ent_branch.add(f"Work Mode: {entities.get('work_mode', 'N/A')}")
            ent_branch.add(f"Seniority: {entities.get('seniority_level', 'N/A')}")
            techs = entities.get("required_technologies", [])
            if techs:
                ent_branch.add(f"Tech Stack: {', '.join(techs[:5])}")

        console.print(tree)

    # Example 3: Incomplete job postings
    console.print(
        "\n[bold yellow]Example 3: Jobs Missing Key Information[/bold yellow]"
    )
    incomplete_jobs = load_jobs_list(
        db_client=db_client,
        where_conditions={"salary": "IS_NULL", "hours_per_week": "IS_NULL"},
    )
    console.print(f"[green]✓ Found {len(incomplete_jobs)} incomplete jobs[/green]")

    # Example 4: High-value contract jobs
    console.print(
        "\n[bold yellow]Example 4: Contract Jobs with Salary (Any Date)[/bold yellow]"
    )
    contract_paid = load_jobs_list(
        db_client=db_client,
        where_conditions={"job_type": "Contract", "salary": "NOT_NULL"},
        include_entities=True,
    )
    console.print(f"[green]✓ Found {len(contract_paid)} paid contract jobs[/green]")

    # Save comprehensive results
    import json

    output_file = OUTPUT_DIR / "advanced_filtering_results.json"
    results = {
        "recent_ft_salary_count": len(recent_ft_salary),
        "remote_linkedin_count": len(remote_with_entities),
        "incomplete_jobs_count": len(incomplete_jobs),
        "paid_contracts_count": len(contract_paid),
        "sample_recent_ft": [dict(j) for j in recent_ft_salary[:3]],
        "sample_remote": [dict(j) for j in remote_with_entities[:3]],
    }
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    console.print(f"\n[spring_green1]✓ Results saved to: {output_file}[/spring_green1]")

    # Summary
    console.print("\n[bold]Advanced Filter Summary:[/bold]")
    console.print(f"  • Recent FT with salary: {len(recent_ft_salary)}")
    console.print(f"  • Remote LinkedIn jobs: {len(remote_with_entities)}")
    console.print(f"  • Incomplete postings: {len(incomplete_jobs)}")
    console.print(f"  • Paid contracts: {len(contract_paid)}")


if __name__ == "__main__":
    main()
