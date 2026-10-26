"""
Using pre-computed embeddings for faster dense search.
Useful when documents don't change frequently.
"""

from pathlib import Path

import numpy as np
from jet.adapters.llama_cpp.config import EMBED_MODEL
from jet.adapters.llama_cpp.embed_utils import embed
from jet.adapters.llama_cpp.rrf_utils import hybrid_rrf_search
from rich.console import Console
from rich.table import Table

console = Console()

OUTPUT_DIR = Path(__file__).parent.parent / "generated" / "hybrid_rrf_search"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DOCUMENTS = [
    "The giant panda is a bear species endemic to China.",
    "Python is a high-level programming language.",
    "Bears are carnivoran mammals of the family Ursidae.",
    "Machine learning is a subset of artificial intelligence.",
    "Pandas eat bamboo and live in mountainous regions.",
]


def main():
    console.print("[bold blue]Hybrid Search with Pre-computed Embeddings[/bold blue]")
    console.print("=" * 60)

    query = "bear species"

    # Pre-compute document embeddings once
    console.print("[dim]Pre-computing embeddings...[/dim]")
    doc_embeddings = embed(DOCUMENTS, model=EMBED_MODEL)
    if isinstance(doc_embeddings, list):
        doc_embeddings = np.array(doc_embeddings, dtype=np.float32)
    console.print(f"[green]Embeddings shape: {doc_embeddings.shape}[/green]")

    # Use pre-computed embeddings in hybrid search
    results = hybrid_rrf_search(
        query=query,
        documents=DOCUMENTS,
        dense_top_n=5,
        sparse_top_n=5,
        final_top_n=3,
        k=60,
        dense_weight=1.0,
        sparse_weight=1.0,
        use_rerank=False,
        sparse_min_score=0.01,
        embed_model=EMBED_MODEL,
        embeddings=doc_embeddings,  # Pass pre-computed embeddings
    )

    table = Table(title="Results", show_header=True, header_style="bold magenta")
    table.add_column("Rank", style="cyan")
    table.add_column("Score", style="yellow")
    table.add_column("Sources", style="blue")
    table.add_column("Text", style="white")

    output_lines = [f"Query: {query}\nUsing pre-computed embeddings\n"]

    for r in results:
        table.add_row(
            str(r["rank"]),
            f"{r['score']:.5f}",
            ", ".join(r["sources"]),
            r["text"][:70],
        )
        output_lines.append(f"#{r['rank']} (score={r['score']:.5f}): {r['text']}")

    console.print(table)

    output_file = OUTPUT_DIR / "04_precomputed_embeddings.txt"
    with open(output_file, "w") as f:
        f.write("\n".join(output_lines))
    console.print(f"[dim]Saved to {output_file}[/dim]")


if __name__ == "__main__":
    main()
