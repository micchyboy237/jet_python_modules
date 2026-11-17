"""
Demo 3: Auto-Centroid Exploration
---------------------------------
Goal: Use MMR to select diverse, representative documents from the entire
dataset without a specific query.

Expected Outcomes:
- Results span multiple domains: Data Science, Web Dev, Core Python, etc.
- The first item is the most "central" document (highest average similarity).
- Subsequent items are chosen for being different from the previous ones.
"""

import json
import shutil
from pathlib import Path

import numpy as np
from jet.adapters.llama_cpp.config import EMBED_DOC_PREFIX, EMBED_MODEL
from jet.adapters.llama_cpp.diversity_utils.mmr import embed, mmr_rerank
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# Output directory setup
OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLES_PATH = Path(__file__).parents[2] / "mocks" / "mmr_samples.json"


def main():
    console.print(
        Panel.fit(
            "[bold]Demo 3: Auto-Centroid Exploration[/bold]\nQuery: None (Auto) | Lambda: 0.7 (Balanced)",
            border_style="magenta",
        )
    )

    # Load Data
    with open(SAMPLES_PATH, "r", encoding="utf-8") as f:
        docs = json.load(f)

    texts = [doc["text"] for doc in docs]

    # Embed
    console.print("[dim]Embedding documents...[/dim]")
    doc_vecs = embed(texts, model=EMBED_MODEL, prefix=EMBED_DOC_PREFIX)

    # Auto Query: Centroid
    query_vec = np.mean(doc_vecs, axis=0)

    # Run MMR
    results = mmr_rerank(query_vec, doc_vecs, top_k=6, lambda_param=0.7)

    # Display
    table = Table(title="MMR Results (λ=0.7, Auto Query)")
    table.add_column("#", style="cyan")
    table.add_column("Video ID")
    table.add_column("Relevance")
    table.add_column("Redundancy")
    table.add_column("Preview")

    for r in results:
        item = docs[r["index"]]
        table.add_row(
            str(r["index"]),
            item["videoId"],
            f"{r['relevance']:.4f}",
            f"{r['redundancy']:.4f}",
            item["text"][:60] + "...",
        )
    console.print(table)

    # Save
    output_file = OUTPUT_DIR / "results.json"
    with open(output_file, "w") as f:
        json.dump({"results": results}, f, indent=2)
    console.print(f"[green]Saved to:[/green] {output_file}")


if __name__ == "__main__":
    main()
