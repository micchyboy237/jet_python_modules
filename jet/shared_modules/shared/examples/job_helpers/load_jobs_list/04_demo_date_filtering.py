"""
Definition Summary:
    Demonstrates filtering jobs by posted date ranges using posted_after and posted_before parameters.
    Useful for finding recent jobs or jobs within a specific time window.
    Filtering is done at the database level using SQL WHERE clauses on posted_date column.

Usage Examples:
    # Get jobs posted in last 7 days
    >>> from datetime import datetime, timedelta
    >>> week_ago = datetime.now() - timedelta(days=7)
    >>> recent_jobs = load_jobs_list(db_client=db_client, posted_after=week_ago)

    # Get jobs from a specific date range
    >>> start = datetime(2026, 9, 1)
    >>> end = datetime(2026, 9, 30)
    >>> sept_jobs = load_jobs_list(db_client=db_client,
    ...                            posted_after=start,
    ...                            posted_before=end)

Span Hierarchies:
    📦 load-jobs-date-filtered (CHAIN)
    │
    └── 🔍 SQL Query Execution
        ├── WHERE posted_date >= %s (ISO format)
        └── WHERE posted_date <= %s (ISO format)
"""

import shutil
from datetime import datetime, timedelta
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
    console.print(Panel.fit("🔍 Demo 04: Date Range Filtering", style="bold cyan"))

    # Initialize database client
    db_client = PgVectorClient(dbname="jobs_db3")

    # Example 1: Jobs from last 7 days
    console.print("\n[bold yellow]Example 1: Jobs Posted in Last 7 Days[/bold yellow]")
    seven_days_ago = datetime.now() - timedelta(days=7)
    recent_jobs = load_jobs_list(db_client=db_client, posted_after=seven_days_ago)
    console.print(f"[green]✓ Found {len(recent_jobs)} jobs from last 7 days[/green]")

    if recent_jobs:
        table1 = Table(
            title="Recent Jobs (Last 7 Days)",
            show_header=True,
            header_style="bold magenta",
        )
        table1.add_column("ID", style="cyan")
        table1.add_column("Title", style="green")
        table1.add_column("Company", style="yellow")
        table1.add_column("Posted Date", style="blue")

        for job in recent_jobs[:5]:
            posted = job.get("posted_date", "")
            if posted and len(posted) > 10:
                posted = posted[:10]
            table1.add_row(
                str(job.get("id", "N/A")),
                str(job.get("title", "N/A"))[:40],
                str(job.get("company", "N/A"))[:30],
                str(posted),
            )
        console.print(table1)

    # Example 2: Jobs from September 2026
    console.print("\n[bold yellow]Example 2: Jobs from September 2026[/bold yellow]")
    sept_start = datetime(2026, 9, 1)
    sept_end = datetime(2026, 9, 30, 23, 59, 59)
    sept_jobs = load_jobs_list(
        db_client=db_client, posted_after=sept_start, posted_before=sept_end
    )
    console.print(f"[green]✓ Found {len(sept_jobs)} jobs from September 2026[/green]")

    # Example 3: Jobs older than 30 days
    console.print("\n[bold yellow]Example 3: Jobs Older Than 30 Days[/bold yellow]")
    thirty_days_ago = datetime.now() - timedelta(days=30)
    older_jobs = load_jobs_list(db_client=db_client, posted_before=thirty_days_ago)
    console.print(f"[green]✓ Found {len(older_jobs)} jobs older than 30 days[/green]")

    # Example 4: Combine date filter with other filters
    console.print(
        "\n[bold yellow]Example 4: Recent Contract Jobs (Last 14 Days)[/bold yellow]"
    )
    fourteen_days_ago = datetime.now() - timedelta(days=14)
    recent_contracts = load_jobs_list(
        db_client=db_client,
        posted_after=fourteen_days_ago,
        where_conditions={"job_type": "Contract"},
    )
    console.print(
        f"[green]✓ Found {len(recent_contracts)} recent contract jobs[/green]"
    )

    # Save results
    import json

    output_file = OUTPUT_DIR / "date_filtered_results.json"
    results = {
        "recent_7_days": len(recent_jobs),
        "september_2026": len(sept_jobs),
        "older_than_30_days": len(older_jobs),
        "recent_contracts": len(recent_contracts),
        "sample_recent": [dict(j) for j in recent_jobs[:3]],
        "sample_sept": [dict(j) for j in sept_jobs[:3]],
    }
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    console.print(f"\n[spring_green1]✓ Results saved to: {output_file}[/spring_green1]")

    # Summary
    console.print("\n[bold]Date Filter Summary:[/bold]")
    console.print(f"  • Last 7 days: {len(recent_jobs)} jobs")
    console.print(f"  • September 2026: {len(sept_jobs)} jobs")
    console.print(f"  • Older than 30 days: {len(older_jobs)} jobs")
    console.print(f"  • Recent contracts (14 days): {len(recent_contracts)} jobs")


if __name__ == "__main__":
    main()
