"""
Demonstrates how fetch_k is calculated for normal and shuffle paths.
Shows the relationship between top_k, diversity, and the required candidate pool size.
"""

import json
import shutil
from pathlib import Path

from jet.adapters.llama_cpp.diversity_utils import (
    compute_fetch_k,
    compute_shuffle_fetch_k,
)
from rich.console import Console
from rich.table import Table

console = Console()

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

console.print("[bold blue]=== Fetch K Calculation Demo ===[/bold blue]")

# Test various top_k and diversity values
test_cases = [
    {"top_k": 5, "diversity": 0.0},
    {"top_k": 5, "diversity": 0.5},
    {"top_k": 5, "diversity": 1.0},
    {"top_k": 10, "diversity": 0.5},
    {"top_k": 20, "diversity": 0.8},
    {"top_k": 50, "diversity": 1.0},
]

table = Table(title="Fetch K Calculations")
table.add_column("Top K", justify="right", style="cyan")
table.add_column("Diversity", justify="right", style="magenta")
table.add_column("Normal Fetch K", justify="right", style="green")
table.add_column("Shuffle Fetch K", justify="right", style="yellow")

results_data = []

for case in test_cases:
    top_k = case["top_k"]
    diversity = case["diversity"]

    normal_fetch = compute_fetch_k(top_k, diversity)
    shuffle_fetch = compute_shuffle_fetch_k(top_k)

    table.add_row(str(top_k), f"{diversity:.1f}", str(normal_fetch), str(shuffle_fetch))

    results_data.append(
        {
            "top_k": top_k,
            "diversity": diversity,
            "normal_fetch_k": normal_fetch,
            "shuffle_fetch_k": shuffle_fetch,
        }
    )

console.print(table)

# Save output
output_file = OUTPUT_DIR / "fetch_k_calculations.json"
output_file.write_text(json.dumps(results_data, indent=2))

console.print(f"\n[bold]Key Insights:[/bold]")
console.print("- Normal fetch K increases with diversity to allow MMR more choices.")
console.print(
    "- Shuffle fetch K is generally larger to ensure variety across shuffles."
)
console.print("- Both are capped to prevent excessive DB queries.")

console.print(f"\n[bold]Saved files:[/]")
for f in sorted(OUTPUT_DIR.iterdir()):
    console.print(f"  📄 [link=file://{f}]{f.name}[/link]")
