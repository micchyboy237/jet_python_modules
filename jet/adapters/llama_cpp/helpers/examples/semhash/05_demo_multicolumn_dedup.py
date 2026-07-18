import json
import os
import shutil
from pathlib import Path

from rich.console import Console
from rich.table import Table
from semhash import SemHash

console = Console()
OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
console.print(f"[bold]Output dir ready:[/bold] [cyan]{OUTPUT_DIR}[/cyan]")

BASE_DIR = os.path.join(os.path.dirname(__file__), "mocks")
sample_data_path = os.path.join(BASE_DIR, "05_data.json")

# Load sample data
with console.status("[bold green]Loading sample data...", spinner="dots"):
    with open(sample_data_path, "r") as f:
        sample_records = json.load(f)
console.print(f"Loaded [bold]{len(sample_records)}[/bold] sample records")

# Build SemHash index with multiple columns
with console.status("[bold green]Building SemHash index...", spinner="dots"):
    semhash = SemHash.from_records(
        records=sample_records, columns=["question", "context"]
    )
console.print("[green]SemHash index built[/green]")

# Deduplicate
with console.status("[bold green]Running deduplication...", spinner="dots"):
    result = semhash.self_deduplicate(threshold=0.9)

# Display results
duplicate_ratio = getattr(result, "duplicate_ratio", "N/A")
table = Table(
    title="Multi-Column Deduplication Results",
    show_header=True,
    header_style="bold magenta",
)
table.add_column("Metric", style="cyan")
table.add_column("Value", style="green")
table.add_row("Input records", str(len(sample_records)))
table.add_row("Deduplicated count", str(len(result.selected)))
table.add_row("Duplicates removed", str(len(sample_records) - len(result.selected)))
table.add_row("Duplicate ratio", str(duplicate_ratio))
console.print(table)

# Save deduplicated records
output_file = OUTPUT_DIR / "cleaned_data.json"
with open(output_file, "w") as f:
    json.dump(result.selected, f, indent=2)
output_uri = output_file.resolve().as_uri()
console.print(
    f"[bold green]Done![/bold green] Saved cleaned output to [link={output_uri}]{output_file}[/link]"
)
