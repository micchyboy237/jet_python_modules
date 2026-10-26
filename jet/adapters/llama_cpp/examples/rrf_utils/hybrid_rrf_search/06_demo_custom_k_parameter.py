"""
Adjusting the RRF smoothing constant k.
Lower k gives more weight to top-ranked items, higher k makes ranking differences less significant.
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
    "The red panda is a small mammal native to the eastern Himalayas.",
    "Bamboo is the primary food source for giant pandas.",
]


def main():
    console.print("[bold blue]Custom RRF Smoothing Constant (k)[/bold blue]")
    console.print("k=10 (gives more weight to top ranks)")
    console.print("=" * 60)

    query = "bamboo eating animals"

    # Lower k = top ranks matter more
    results = hybrid_rrf_search(
        query=query,
        documents=DOCUMENTS,
        dense_top_n=5,
        sparse_top_n=5,
        final_top_n=3,
        k=10,  # Lower than default 60
        dense_weight=1.0,
        sparse_weight=1.0,
        use_rerank=False,
        sparse_min_score=0.01,
        embed_model=EMBED_MODEL,
    )

    table = Table(title="Results (k=10)", show_header=True, header_style="bold magenta")
    table.add_column("Rank", style="cyan")
    table.add_column("Score", style="yellow")
    table.add_column("Sources", style="blue")
    table.add_column("Text", style="white")

    output_lines = [f"Query: {query}\nk=10 (top-rank emphasis)\n"]

    for r in results:
        table.add_row(
            str(r["rank"]),
            f"{r['score']:.5f}",
            ", ".join(r["sources"]),
            r["text"][:70],
        )
        output_lines.append(f"#{r['rank']} (score={r['score']:.5f}): {r['text']}")

    console.print(table)

    output_file = OUTPUT_DIR / "06_custom_k.txt"
    with open(output_file, "w") as f:
        f.write("\n".join(output_lines))
    console.print(f"[dim]Saved to {output_file}[/dim]")


if __name__ == "__main__":
    main()
