"""
Demonstrates weighted fusion where dense and sparse retrievers have different importance.
Use this when one retrieval method is more reliable than the other.
"""

from pathlib import Path

from jet.adapters.llama_cpp.config import EMBED_MODEL
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
    console.print("[bold blue]Weighted Hybrid Fusion[/bold blue]")
    console.print("Dense weight=2.0, Sparse weight=0.5")
    console.print("=" * 60)

    query = "programming languages"

    # Emphasize semantic similarity over keyword matching
    results = hybrid_rrf_search(
        query=query,
        documents=DOCUMENTS,
        dense_top_n=5,
        sparse_top_n=5,
        final_top_n=3,
        k=60,
        dense_weight=2.0,  # Give 2x importance to dense results
        sparse_weight=0.5,  # Give 0.5x importance to BM25 results
        use_rerank=False,
        sparse_min_score=0.01,
        embed_model=EMBED_MODEL,
    )

    table = Table(
        title="Weighted Results", show_header=True, header_style="bold magenta"
    )
    table.add_column("Rank", style="cyan")
    table.add_column("Score", style="yellow")
    table.add_column("Sources", style="blue")
    table.add_column("Text", style="white")

    output_lines = [f"Query: {query}\nWeights: dense=2.0, sparse=0.5\n"]

    for r in results:
        table.add_row(
            str(r["rank"]),
            f"{r['score']:.5f}",
            ", ".join(r["sources"]),
            r["text"][:70],
        )
        output_lines.append(f"#{r['rank']} (score={r['score']:.5f}): {r['text']}")

    console.print(table)

    output_file = OUTPUT_DIR / "02_weighted_fusion.txt"
    with open(output_file, "w") as f:
        f.write("\n".join(output_lines))
    console.print(f"[dim]Saved to {output_file}[/dim]")


if __name__ == "__main__":
    main()
