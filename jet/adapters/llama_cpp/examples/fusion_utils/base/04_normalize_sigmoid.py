"""Sigmoid Normalization Demo.

Real-world use case: Converting raw cross-encoder reranker logits into
interpretable 0–1 relevance scores for display in UIs or threshold-based
filtering. Temperature parameter controls how aggressively scores cluster
toward extremes vs. center, enabling calibration to your domain.
"""

import json
import shutil
from pathlib import Path

from jet.adapters.llama_cpp.fusion_utils import normalize_sigmoid
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SETTINGS = {
    "normalizer": "sigmoid",
    "temperatures": [0.5, 1.0, 2.0],
    "description": "Effect of temperature on sigmoid normalization of raw reranker logits",
}

INPUTS = {
    "raw_scores": [-3.0, -1.0, 0.0, 0.5, 1.5, 3.0, 5.0],
    "labels": [
        "Clearly irrelevant",
        "Likely irrelevant",
        "Neutral",
        "Slightly relevant",
        "Relevant",
        "Highly relevant",
        "Definitive match",
    ],
}

console.print(Panel("📈 Sigmoid Normalization Demo", style="bold cyan"))

console.print("\n[bold]Raw Reranker Logits:[/bold]")
raw_table = Table(show_header=True, header_style="bold magenta")
raw_table.add_column("Label", width=22)
raw_table.add_column("Raw Score", justify="right", width=12)
for label, score in zip(INPUTS["labels"], INPUTS["raw_scores"]):
    raw_table.add_row(label, f"{score:.1f}")
console.print(raw_table)

all_normalized = {}
for temp in SETTINGS["temperatures"]:
    all_normalized[str(temp)] = normalize_sigmoid(
        INPUTS["raw_scores"], temperature=temp
    ).tolist()

console.print("\n[bold]Normalized Scores by Temperature:[/bold]")
comp_table = Table(show_header=True, header_style="bold green")
comp_table.add_column("Raw", justify="right", width=8)
comp_table.add_column("Label", width=20)
for temp in SETTINGS["temperatures"]:
    comp_table.add_column(f"T={temp}", justify="right", width=10)
for i, (label, raw) in enumerate(zip(INPUTS["labels"], INPUTS["raw_scores"])):
    row = [f"{raw:.1f}", label]
    for temp in SETTINGS["temperatures"]:
        row.append(f"{all_normalized[str(temp)][i]:.4f}")
    comp_table.add_row(*row)
console.print(comp_table)

console.print(
    "\n[yellow]💡 Lower temperature → sharper discrimination between relevant/irrelevant[/yellow]"
)
console.print(
    "[yellow]   Higher temperature → softer gradient, more scores near 0.5[/yellow]"
)

outputs = {
    "normalized_by_temperature": all_normalized,
    "interpretation_guide": {
        "0.5": "neutral boundary",
        ">0.7": "high relevance",
        "<0.3": "low relevance",
    },
}

with open(OUTPUT_DIR / "settings.json", "w") as f:
    json.dump(SETTINGS, f, indent=2)
with open(OUTPUT_DIR / "inputs.json", "w") as f:
    json.dump(INPUTS, f, indent=2)
with open(OUTPUT_DIR / "outputs.json", "w") as f:
    json.dump(outputs, f, indent=2)

console.print(f"\n[dim]Outputs saved to {OUTPUT_DIR}[/dim]")
