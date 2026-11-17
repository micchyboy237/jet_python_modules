"""
Demo 2: Pure Relevance Baseline
-------------------------------
Goal: Show standard Top-K behavior where diversity is ignored (Lambda = 1.0).

Expected Outcomes:
- Rank 1-3: All three "Python List" explanations appear at the top.
- High redundancy scores between the top items.
- Useful as a comparison to Demo 1 to see what MMR filters out.
"""

import json
import shutil
from pathlib import Path

from jet.adapters.llama_cpp.config import (
    EMBED_DOC_PREFIX,
    EMBED_MODEL,
    EMBED_QUERY_PREFIX,
)
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
            "[bold]Demo 2: Pure Relevance Baseline[/bold]\nQuery: 'python lists' | Lambda: 1.0 (No Diversity)",
            border_style="yellow",
        )
    )

    # Load Data
    with open(SAMPLES_PATH, "r", encoding="utf-8") as f:
        docs = json.load(f)

    texts = [doc["text"] for doc in docs]

    # Embed
    console.print("[dim]Embedding documents...[/dim]")
    doc_vecs = embed(texts, model=EMBED_MODEL, prefix=EMBED_DOC_PREFIX)
    query_vec = embed("python lists", model=EMBED_MODEL, prefix=EMBED_QUERY_PREFIX)

    # Run MMR
    results = mmr_rerank(query_vec, doc_vecs, top_k=5, lambda_param=1.0)

    # Display
    table = Table(title="MMR Results (λ=1.0)")
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
