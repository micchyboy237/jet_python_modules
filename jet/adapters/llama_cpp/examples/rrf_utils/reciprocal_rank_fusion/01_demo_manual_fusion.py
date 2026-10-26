"""
Basic manual RRF fusion of two ranked lists.
Shows how to fuse results from different retrieval systems.
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
]


def main():
    console.print("[bold blue]Manual RRF Fusion of Two Lists[/bold blue]")
    console.print("=" * 60)

    # Simulate results from two different retrievers
    dense_results = [
        {"index": 0, "score": 0.92, "text": DOCUMENTS[0]},
        {"index": 4, "score": 0.85, "text": DOCUMENTS[4]},
        {"index": 2, "score": 0.71, "text": DOCUMENTS[2]},
    ]

    bm25_results = [
        {"index": 0, "score": 12.4, "text": DOCUMENTS[0]},
        {"index": 4, "score": 8.1, "text": DOCUMENTS[4]},
        {"index": 2, "score": 5.3, "text": DOCUMENTS[2]},
    ]

    fused = reciprocal_rank_fusion(
        ranked_lists=[dense_results, bm25_results],
        k=60,
        weights=[1.0, 1.0],
        list_names=["dense", "bm25"],
        top_n=5,
        id_key="index",
    )

    table = Table(title="Fused Results", show_header=True, header_style="bold magenta")
    table.add_column("Rank", style="cyan")
    table.add_column("RRF Score", style="yellow")
    table.add_column("Sources", style="blue")
    table.add_column("Original Ranks", style="dim")
    table.add_column("Text", style="white")

    output_lines = ["Manual RRF Fusion\n"]

    for r in fused:
        table.add_row(
            str(r["rank"]),
            f"{r['score']:.5f}",
            ", ".join(r["sources"]),
            str(r["original_ranks"]),
            r["text"][:60],
        )
        output_lines.append(
            f"#{r['rank']} (RRF={r['score']:.5f}): {r['text']}\n"
            f"  Sources: {r['sources']}, Original ranks: {r['original_ranks']}"
        )

    console.print(table)

    output_file = OUTPUT_DIR / "01_manual_fusion.txt"
    with open(output_file, "w") as f:
        f.write("\n".join(output_lines))
    console.print(f"[dim]Saved to {output_file}[/dim]")


if __name__ == "__main__":
    main()
