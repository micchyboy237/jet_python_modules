"""
Demo 1: Boilerplate & Near-Lexical Dedup
----------------------------------------
Goal: Show how MinHash LSH removes near-identical textual duplicates
(e.g., legal disclaimers, repeated headers) before semantic processing.
Expected Outcomes:
- "Exact" dedup removes byte-identical strings.
- "Near-lexical" dedup removes strings with high Jaccard similarity
  (e.g., same words, slightly different order or punctuation).
- Semantic dedup then works on a cleaner, more diverse set.
"""

import shutil
from pathlib import Path

from jet.adapters.llama_cpp.diversity_utils.fast_mmr import (
    DedupConfig,
    diversity_preserving_dedup,
)
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()
OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Sample data with heavy boilerplate repetition
SAMPLE_TEXTS = [
    "Terms of Service: You agree to our policies. Version 1.0",
    "Terms of Service: You agree to our policies. Version 1.1",  # Near-duplicate
    "Terms of Service: You agree to our policies. Version 1.0",  # Exact duplicate
    "Privacy Policy: We collect your data. Copyright 2023.",
    "Privacy Policy: We collect your data. Copyright 2024.",  # Near-duplicate
    "Introduction to Python programming.",
    "Advanced Machine Learning techniques.",
    "Data structures and algorithms in C++.",
]


def main():
    console.print(
        Panel.fit(
            "[bold]Demo 1: Boilerplate Removal[/bold]\nUsing MinHash LSH for near-lexical dedup",
            border_style="cyan",
        )
    )

    # Configure with a high threshold to catch slight variations
    cfg = DedupConfig(
        minhash_threshold=0.8,
        minhash_num_perm=128,
        n_clusters=3,
        semantic_sim_threshold=0.9,
    )

    console.print(f"[dim]Input texts: {len(SAMPLE_TEXTS)}[/dim]")
    for i, t in enumerate(SAMPLE_TEXTS):
        console.print(f"  [{i}] {t[:50]}...")

    console.print("\n[dim]Running 3-layer deduplication...[/dim]")
    kept_indices, stats, _ = diversity_preserving_dedup(SAMPLE_TEXTS, cfg)

    table = Table(title="Deduplication Stats")
    table.add_column("Stage", style="cyan")
    table.add_column("Count", justify="right")
    for stage, count in stats.items():
        table.add_row(stage.replace("_", " ").title(), str(count))
    console.print(table)

    console.print("\n[bold green]Kept Indices (Unique Content):[/bold green]")
    for idx in kept_indices:
        console.print(f"  [{idx}] {SAMPLE_TEXTS[idx]}")


if __name__ == "__main__":
    main()
