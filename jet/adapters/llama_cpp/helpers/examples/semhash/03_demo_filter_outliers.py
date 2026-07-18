"""
03_demo_filter_outliers.py

Demonstrates SemHash.self_filter_outliers(): ranking every record in a
fitted index by its average similarity to its own nearest neighbors, then
splitting the dataset into:
  - selected (inliers): records that resemble other records in the dataset
  - filtered (outliers): records that stand apart from everything else

This is the self-contained analog of self_find_representative — instead of
picking a diverse representative subset, it flags the records that don't fit
in with the rest.
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
sample_data_path = os.path.join(BASE_DIR, "03_data.json")

with console.status("[bold green]Loading sample data...", spinner="dots"):
    with open(sample_data_path, "r") as f:
        sample_records = json.load(f)
console.print(f"Loaded [bold]{len(sample_records)}[/bold] sample records")

with console.status("[bold green]Building SemHash index...", spinner="dots"):
    semhash = SemHash.from_records(records=sample_records, columns=["text"])
console.print("[green]SemHash index built[/green]")

OUTLIER_PERCENTAGE = 0.2  # bottom 20% least-similar records flagged as outliers

with console.status("[bold green]Filtering outliers...", spinner="dots"):
    result = semhash.self_filter_outliers(outlier_percentage=OUTLIER_PERCENTAGE)

inlier_count = len(result.selected)
outlier_count = len(result.filtered)

table = Table(
    title="Outlier Filtering Results",
    show_header=True,
    header_style="bold magenta",
)
table.add_column("Metric", style="cyan")
table.add_column("Value", style="green")
table.add_row("Input records", str(len(sample_records)))
table.add_row("Outlier percentage", f"{OUTLIER_PERCENTAGE:.0%}")
table.add_row("Inliers (selected)", str(inlier_count))
table.add_row("Outliers (filtered)", str(outlier_count))
console.print(table)

# Show the actual outlier texts + their similarity scores for quick sanity-check
outlier_table = Table(
    title="Detected Outliers",
    show_header=True,
    header_style="bold red",
)
outlier_table.add_column("Text", style="white")
outlier_table.add_column("Avg similarity score", style="yellow")
for record, score in zip(result.filtered, result.scores_filtered):
    outlier_table.add_row(record["text"], f"{score:.4f}")
console.print(outlier_table)

inliers_file = OUTPUT_DIR / "inliers.json"
outliers_file = OUTPUT_DIR / "outliers.json"

with open(inliers_file, "w") as f:
    json.dump(result.selected, f, indent=2)
with open(outliers_file, "w") as f:
    json.dump(result.filtered, f, indent=2)

console.print(
    f"[bold green]Done![/bold green] "
    f"Saved inliers to [link={inliers_file.resolve().as_uri()}]{inliers_file}[/link], "
    f"outliers to [link={outliers_file.resolve().as_uri()}]{outliers_file}[/link]"
)
