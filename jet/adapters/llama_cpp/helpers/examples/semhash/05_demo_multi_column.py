"""
05_demo_multi_column.py

Demonstrates SemHash with MULTIPLE columns, using the default SemHash model
(no custom encoder). Records here have two columns: "title" and "text".

How multi-column featurization works (see semhash/utils.py: featurize()):
  - Each column is encoded SEPARATELY: model.encode(titles), model.encode(texts)
  - The resulting embeddings are concatenated side-by-side (axis=1)
  - So a record's final vector is [title_embedding | text_embedding]

This means similarity comparisons take BOTH columns into account jointly —
two records only rank as near-duplicates if both their titles AND their
texts are similar.
"""

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

with console.status("[bold green]Loading sample data...", spinner="dots"):
    with open(sample_data_path, "r") as f:
        sample_records = json.load(f)
console.print(f"Loaded [bold]{len(sample_records)}[/bold] sample records")

COLUMNS = ["title", "text"]
console.log(f"Using columns: {COLUMNS}")

with console.status(
    "[bold green]Building SemHash index (multi-column)...", spinner="dots"
):
    semhash = SemHash.from_records(records=sample_records, columns=COLUMNS)
console.print(f"[green]SemHash index built[/green] using columns {COLUMNS}")

with console.status("[bold green]Finding representative samples...", spinner="dots"):
    result = semhash.self_find_representative(diversity=0.5)

representative_count = len(result.selected)
table = Table(
    title="Representative Samples Results (multi-column: title + text)",
    show_header=True,
    header_style="bold magenta",
)
table.add_column("Metric", style="cyan")
table.add_column("Value", style="green")
table.add_row("Input records", str(len(sample_records)))
table.add_row("Columns used", ", ".join(COLUMNS))
table.add_row("Representative count", str(representative_count))
table.add_row(
    "Reduction ratio", f"{1 - (representative_count / len(sample_records)):.2%}"
)
console.print(table)

# Show selected representatives with both columns visible
repr_table = Table(title="Selected Representatives", header_style="bold cyan")
repr_table.add_column("Title", style="white")
repr_table.add_column("Text", style="white")
for record in result.selected:
    repr_table.add_row(record["title"], record["text"])
console.print(repr_table)

output_file = OUTPUT_DIR / "representative_data.json"
with open(output_file, "w") as f:
    json.dump(result.selected, f, indent=2)

output_uri = output_file.resolve().as_uri()
console.print(
    f"[bold green]Done![/bold green] Saved representative output to [link={output_uri}]{output_file}[/link]"
)
