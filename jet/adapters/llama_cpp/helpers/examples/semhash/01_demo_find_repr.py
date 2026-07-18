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
sample_data_path = os.path.join(BASE_DIR, "01_data.json")

# Load sample data
with console.status("[bold green]Loading sample data...", spinner="dots"):
    with open(sample_data_path, "r") as f:
        sample_records = json.load(f)
console.print(f"Loaded [bold]{len(sample_records)}[/bold] sample records")

# Build SemHash index
with console.status("[bold green]Building SemHash index...", spinner="dots"):
    semhash = SemHash.from_records(records=sample_records, columns=["text"])
console.print("[green]SemHash index built[/green]")

# Find representative samples
with console.status("[bold green]Finding representative samples...", spinner="dots"):
    result = semhash.self_find_representative(diversity=0.5, selection_size=10)

# Display results
representative_count = len(result.selected)
table = Table(
    title="Representative Samples Results",
    show_header=True,
    header_style="bold magenta",
)
table.add_column("Metric", style="cyan")
table.add_column("Value", style="green")
table.add_row("Input records", str(len(sample_records)))
table.add_row("Representative count", str(representative_count))
table.add_row(
    "Reduction ratio", f"{1 - (representative_count / len(sample_records)):.2%}"
)

console.print(table)

# Save representative samples
output_file = OUTPUT_DIR / "representative_data.json"
with open(output_file, "w") as f:
    json.dump(result.selected, f, indent=2)
output_uri = output_file.resolve().as_uri()
console.print(
    f"[bold green]Done![/bold green] Saved representative output to [link={output_uri}]{output_file}[/link]"
)
