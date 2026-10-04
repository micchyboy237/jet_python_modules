"""
Definition Summary:
    Demonstrates performance considerations and best practices when using load_jobs_list().
    Shows how to handle large datasets, combine with other helpers, and optimize queries.
    Includes examples of batch processing and memory-efficient iteration.

Usage Examples:
    # Load only jobs with specific criteria for efficiency
    >>> jobs = load_jobs_list(
    ...     db_client=db_client,
    ...     where_conditions={"domain": "linkedin.com"},
    ...     include_entities=False  # Skip entity join if not needed
    ... )

    # Process in batches
    >>> all_jobs = load_jobs_list(db_client=db_client)
    >>> for i in range(0, len(all_jobs), 100):
    ...     batch = all_jobs[i:i+100]
    ...     process_batch(batch)

Span Hierarchies:
    📦 load-jobs-performance-demo (CHAIN)
    │
    ├── 🔍 SQL Query Execution
    │   └── Optimized with WHERE clauses and selective column loading
    │
    └── 🔄 Data Processing
        ├── Batch iteration
        └── Memory-efficient handling
"""

import shutil
from pathlib import Path

from jet.db.postgres.pgvector import PgVectorClient
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from shared.job_helpers import load_jobs_list

console = Console()

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    console.print(
        Panel.fit("🔍 Demo 06: Performance & Best Practices", style="bold cyan")
    )

    # Initialize database client
    db_client = PgVectorClient(dbname="jobs_db3")

    # Example 1: Efficient loading without entities
    console.print(
        "\n[bold yellow]Example 1: Fast Loading (No Entity Join)[/bold yellow]"
    )
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading jobs...", total=None)
        jobs_no_entities = load_jobs_list(db_client=db_client, include_entities=False)
        progress.update(task, description=f"Loaded {len(jobs_no_entities)} jobs")

    console.print(f"[green]✓ Loaded {len(jobs_no_entities)} jobs (fast mode)[/green]")

    # Example 2: Selective loading with filters
    console.print(
        "\n[bold yellow]Example 2: Selective Loading (LinkedIn Only)[/bold yellow]"
    )
    linkedin_jobs = load_jobs_list(
        db_client=db_client,
        where_conditions={"domain": "linkedin.com"},
        include_entities=False,
    )
    console.print(f"[green]✓ Loaded {len(linkedin_jobs)} LinkedIn jobs[/green]")

    # Example 3: Batch processing simulation
    console.print("\n[bold yellow]Example 3: Batch Processing Pattern[/bold yellow]")
    BATCH_SIZE = 100
    total_jobs = len(jobs_no_entities)

    console.print(f"Processing {total_jobs} jobs in batches of {BATCH_SIZE}...")

    batch_stats = []
    for i in range(0, min(total_jobs, 300), BATCH_SIZE):  # Limit to 300 for demo
        batch = jobs_no_entities[i : i + BATCH_SIZE]

        # Simulate processing
        titles_in_batch = [j.get("title", "") for j in batch]
        companies_in_batch = set(
            j.get("company", "") for j in batch if j.get("company")
        )

        batch_stats.append(
            {
                "batch_num": i // BATCH_SIZE + 1,
                "count": len(batch),
                "unique_companies": len(companies_in_batch),
            }
        )

    table = Table(
        title="Batch Processing Stats", show_header=True, header_style="bold magenta"
    )
    table.add_column("Batch #", style="cyan")
    table.add_column("Jobs in Batch", style="green")
    table.add_column("Unique Companies", style="yellow")

    for stat in batch_stats:
        table.add_row(
            str(stat["batch_num"]), str(stat["count"]), str(stat["unique_companies"])
        )

    console.print(table)

    # Example 4: Memory-efficient filtering
    console.print("\n[bold yellow]Example 4: Post-Load Filtering[/bold yellow]")

    # Load all jobs once
    all_jobs = load_jobs_list(db_client=db_client, include_entities=False)

    # Filter in Python (useful for complex logic not expressible in SQL)
    high_salary_jobs = [
        j
        for j in all_jobs
        if j.get("salary")
        and any(x in str(j["salary"]).lower() for x in ["1000", "2000", "3000"])
    ]

    console.print(
        f"[green]✓ Found {len(high_salary_jobs)} potentially high-salary jobs[/green]"
    )

    # Example 5: Combining with other helpers
    console.print(
        "\n[bold yellow]Example 5: Integration with Other Helpers[/bold yellow]"
    )

    # Get a sample job ID
    if all_jobs:
        sample_job_id = all_jobs[0]["id"]
        console.print(f"Sample Job ID: {sample_job_id}")

        # Could use this ID with other helpers like:
        # - load_job_metadata(job_id, db_client)
        # - load_job_entities(job_id, db_client)
        # - load_job_summary(job_id, db_client)

        console.print(
            "[dim]These can be used for detailed lookups after bulk loading[/dim]"
        )

    # Save performance stats
    import json

    output_file = OUTPUT_DIR / "performance_stats.json"
    results = {
        "total_jobs_loaded": len(jobs_no_entities),
        "linkedin_jobs": len(linkedin_jobs),
        "batch_processing_stats": batch_stats,
        "high_salary_candidates": len(high_salary_jobs),
        "sample_job_ids": [j["id"] for j in all_jobs[:5]],
    }
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    console.print(f"\n[spring_green1]✓ Results saved to: {output_file}[/spring_green1]")

    # Best practices summary
    console.print("\n[bold]Best Practices:[/bold]")
    console.print(
        "  • Use [cyan]include_entities=False[/cyan] when entities aren't needed"
    )
    console.print(
        "  • Apply [cyan]where_conditions[/cyan] to reduce result set at DB level"
    )
    console.print("  • Use [cyan]posted_after/before[/cyan] for time-bounded queries")
    console.print("  • Process large datasets in [cyan]batches[/cyan] to manage memory")
    console.print(
        "  • Combine with [cyan]load_job_*[/cyan] helpers for detailed lookups"
    )


if __name__ == "__main__":
    main()
