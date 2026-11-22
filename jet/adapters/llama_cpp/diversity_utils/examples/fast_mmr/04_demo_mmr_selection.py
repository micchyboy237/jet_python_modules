"""
Demo 4: MMR Selection on Deduplicated Data
------------------------------------------
Goal: Select a diverse subset from the cleaned data using MMR.
Expected Outcomes:
- Final output is small, diverse, and representative.
- Demonstrates the end-to-end pipeline: Dedup -> Embed -> MMR.
"""

import shutil
from pathlib import Path

from jet.adapters.llama_cpp.diversity_utils.fast_mmr import (
    DedupConfig,
    get_diverse_subset,
)
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()
OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_TEXTS = [
    "Python is a programming language.",
    "Python is a snake.",
    "Java is a programming language.",
    "Java is an island.",
    "C++ is used for system programming.",
    "Ruby is known for web development.",
    "Go is designed for concurrency.",
    "Rust focuses on memory safety.",
    "Python is great for AI.",  # Semantic duplicate of #1
    "Java is popular in enterprise.",  # Semantic duplicate of #3
]


def main():
    console.print(
        Panel.fit(
            "[bold]Demo 4: MMR Selection[/bold]\nSelecting Top-3 Diverse Items",
            border_style="green",
        )
    )

    cfg = DedupConfig(minhash_threshold=0.8, n_clusters=4, semantic_sim_threshold=0.85)

    k = 3
    console.print(
        f"[dim]Input: {len(SAMPLE_TEXTS)} texts | Target: {k} diverse items[/dim]"
    )

    selected_indices, selected_texts, stats = get_diverse_subset(
        texts=SAMPLE_TEXTS,
        k=k,
        cfg=cfg,
        lambda_param=0.5,  # Balanced relevance and diversity
    )

    console.print(f"\n[bold]Pipeline Stats:[/bold]")
    for stage, count in stats.items():
        console.print(f"  {stage}: {count}")

    table = Table(title=f"Top-{k} Diverse Results")
    table.add_column("Rank", style="cyan")
    table.add_column("Original Index")
    table.add_column("Text")

    for i, (idx, text) in enumerate(zip(selected_indices, selected_texts)):
        table.add_row(str(i + 1), str(idx), text)

    console.print(table)


if __name__ == "__main__":
    main()
