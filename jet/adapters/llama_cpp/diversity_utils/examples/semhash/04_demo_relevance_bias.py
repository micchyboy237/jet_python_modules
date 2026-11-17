"""
Demonstrates the effect of relevance_bias in weighted sampling.
Higher bias favors top-scored items, lower bias favors variety.
"""

import json
import shutil
from pathlib import Path

from jet.adapters.llama_cpp.diversity_utils import sample_candidate_pool
from rich.console import Console
from rich.table import Table

console = Console()

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Mock data: Candidates with clear score hierarchy
candidates = [
    {"id": "high_1", "score": 0.95},
    {"id": "high_2", "score": 0.92},
    {"id": "med_1", "score": 0.70},
    {"id": "med_2", "score": 0.65},
    {"id": "low_1", "score": 0.40},
    {"id": "low_2", "score": 0.35},
    {"id": "low_3", "score": 0.30},
    {"id": "low_4", "score": 0.25},
]

console.print("[bold blue]=== Relevance Bias Demo ===[/bold blue]")
console.print(f"Candidate pool size: {len(candidates)}")
console.print(f"Sample size: 4")
console.print(f"Guaranteed top: 1 (Always keeps the best match)")

bias_levels = [0.5, 1.0, 2.0]
all_samples = {}

table = Table(title="Sampling Results by Relevance Bias")
table.add_column("Bias", justify="right", style="cyan")
table.add_column("Selected IDs", style="green")

for bias in bias_levels:
    # Use fixed seed for reproducibility in demo
    sampled = sample_candidate_pool(
        results=candidates,
        sample_size=4,
        seed=42,
        relevance_bias=bias,
        guaranteed_top=1,
    )

    selected_ids = [item["id"] for item in sampled]
    all_samples[f"bias_{bias}"] = selected_ids

    table.add_row(f"{bias:.1f}", ", ".join(selected_ids))

    console.print(f"\n[bold yellow]Bias: {bias}[/bold yellow]")
    for item in sampled:
        console.print(f"  - {item['id']} (Score: {item['score']:.2f})")

console.print("\n")
console.print(table)

# Save output
output_file = OUTPUT_DIR / "relevance_bias_samples.json"
output_file.write_text(json.dumps(all_samples, indent=2))

console.print(f"\n[bold]Observation:[/bold]")
console.print("- Low bias (0.5) picks more lower-scored items for variety.")
console.print("- High bias (2.0) sticks closer to the top-scoring items.")
console.print("- The top item is always included due to guaranteed_top=1.")

console.print(f"\n[bold]Saved files:[/]")
for f in sorted(OUTPUT_DIR.iterdir()):
    console.print(f"  📄 [link=file://{f}]{f.name}[/link]")
