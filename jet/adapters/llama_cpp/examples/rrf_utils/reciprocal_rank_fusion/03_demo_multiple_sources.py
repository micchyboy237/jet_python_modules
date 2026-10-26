"""
RRF fusion of three or more ranked lists from different sources.
Demonstrates multi-source fusion capability.
"""

from pathlib import Path

from jet.adapters.llama_cpp.rrf_utils import reciprocal_rank_fusion
from rich.console import Console
from rich.table import Table

console = Console()

OUTPUT_DIR = Path(__file__).parent.parent / "generated" / "reciprocal_rank_fusion"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DOCUMENTS = [
    "The giant panda is a bear species endemic to China.",
    "Python is a high-level programming language.",
    "Bears are carnivoran mammals of the family Ursidae.",
    "Machine learning is a subset of artificial intelligence.",
    "Pandas eat bamboo and live in mountainous regions.",
    "The red panda is a small mammal native to the eastern Himalayas.",
]


def main():
    console.print("[bold blue]Multi-Source RRF Fusion (3 lists)[/bold blue]")
    console.print("=" * 60)

    # Three different retrieval systems
    dense_results = [
        {"index": 0, "score": 0.92, "text": DOCUMENTS[0]},
        {"index": 4, "score": 0.85, "text": DOCUMENTS[4]},
        {"index": 5, "score": 0.70, "text": DOCUMENTS[5]},
    ]

    bm25_results = [
        {"index": 0, "score": 12.4, "text": DOCUMENTS[0]},
        {"index": 4, "score": 8.1, "text": DOCUMENTS[4]},
        {"index": 2, "score": 5.3, "text": DOCUMENTS[2]},
    ]

    keyword_results = [
        {"index": 4, "score": 0.95, "text": DOCUMENTS[4]},
        {"index": 0, "score": 0.88, "text": DOCUMENTS[0]},
        {"index": 6, "score": 0.60, "text": "Bamboo forests are panda habitats."},
    ]

    fused = reciprocal_rank_fusion(
        ranked_lists=[dense_results, bm25_results, keyword_results],
        k=60,
        weights=[1.0, 1.0, 0.8],
        list_names=["dense", "bm25", "keyword"],
        top_n=5,
        id_key="index",
    )

    table = Table(
        title="Multi-Source Fused Results",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Rank", style="cyan")
    table.add_column("RRF Score", style="yellow")
    table.add_column("Sources", style="blue")
    table.add_column("Original Ranks", style="dim")
    table.add_column("Text", style="white")

    output_lines = ["Multi-Source RRF Fusion (3 lists)\n"]

    for r in fused:
        table.add_row(
            str(r["rank"]),
            f"{r['score']:.5f}",
            ", ".join(r["sources"]),
            str(r["original_ranks"]),
            r["text"][:50],
        )
        output_lines.append(
            f"#{r['rank']} (RRF={r['score']:.5f}): {r['text']}\n"
            f"  Sources: {r['sources']}"
        )

    console.print(table)

    output_file = OUTPUT_DIR / "03_multiple_sources.txt"
    with open(output_file, "w") as f:
        f.write("\n".join(output_lines))
    console.print(f"[dim]Saved to {output_file}[/dim]")


if __name__ == "__main__":
    main()
