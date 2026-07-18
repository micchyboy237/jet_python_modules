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
sample_data_path = os.path.join(BASE_DIR, "08_data.json")

# Load train and test data
with console.status("[bold green]Loading datasets...", spinner="dots"):
    with open(sample_data_path, "r") as f:
        data = json.load(f)
    train_records = data["train"]
    test_records = data["test"]
console.print(
    f"Loaded [bold]{len(train_records)}[/bold] train records and [bold]{len(test_records)}[/bold] test records"
)

# Build SemHash index for train
with console.status("[bold green]Building SemHash index for train...", spinner="dots"):
    train_semhash = SemHash.from_records(records=train_records, columns=["text"])
console.print("[green]Train SemHash index built[/green]")

# Deduplicate test against train
with console.status("[bold green]Deduplicating test against train...", spinner="dots"):
    result = train_semhash.deduplicate(records=test_records, threshold=0.9)

# Display results
table = Table(
    title="Cross-Dataset Deduplication Results",
    show_header=True,
    header_style="bold magenta",
)
table.add_column("Metric", style="cyan")
table.add_column("Value", style="green")
table.add_row("Test records", str(len(test_records)))
table.add_row("Deduplicated test records", str(len(result.selected)))
table.add_row("Duplicates removed", str(len(test_records) - len(result.selected)))
console.print(table)

# Save deduplicated test records
output_file = OUTPUT_DIR / "deduplicated_test.json"
with open(output_file, "w") as f:
    json.dump(result.selected, f, indent=2)
output_uri = output_file.resolve().as_uri()
console.print(
    f"[bold green]Done![/bold green] Saved deduplicated test output to [link={output_uri}]{output_file}[/link]"
)
