"""
Demonstrates top_n parameter for limiting the number of returned results.
Shows how RRF can return fewer results than the total unique documents.
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
    "Bamboo is the primary food source for giant pandas.",
]


def main():
    console.print("[bold blue]RRF with top_n Limiting[/bold blue]")
    console.print("=" * 60)

    dense_results = [
        {"index": i, "score": 0.9 - i * 0.1, "text": DOCUMENTS[i]} for i in range(7)
    ]

    bm25_results = [
        {"index": i, "score": 10.0 - i * 1.5, "text": DOCUMENTS[i]} for i in range(7)
    ]

    # Fuse all documents but only return top 3
    fused_all = reciprocal_rank_fusion(
        ranked_lists=[dense_results, bm25_results],
        k=60,
        weights=[1.0, 1.0],
        list_names=["dense", "bm25"],
        top_n=None,  # Return all fused results
        id_key="index",
    )

    console.print(f"\n[yellow]Total unique documents fused: {len(fused_all)}[/yellow]")

    # Now limit to top 3
    fused_limited = reciprocal_rank_fusion(
        ranked_lists=[dense_results, bm25_results],
        k=60,
        weights=[1.0, 1.0],
        list_names=["dense", "bm25"],
        top_n=3,  # Only return top 3
        id_key="index",
    )

    console.print(f"[yellow]Limited to top: {len(fused_limited)}[/yellow]\n")

    table = Table(
        title="Top-3 Limited Results", show_header=True, header_style="bold magenta"
    )
    table.add_column("Rank", style="cyan")
    table.add_column("RRF Score", style="yellow")
    table.add_column("Sources", style="blue")
    table.add_column("Text", style="white")

    output_lines = [
        f"Total fused: {len(fused_all)}, Limited to: {len(fused_limited)}\n"
    ]

    for r in fused_limited:
        table.add_row(
            str(r["rank"]),
            f"{r['score']:.5f}",
            ", ".join(r["sources"]),
            r["text"][:60],
        )
        output_lines.append(f"#{r['rank']} (RRF={r['score']:.5f}): {r['text']}")

    console.print(table)

    output_file = OUTPUT_DIR / "05_top_n_limiting.txt"
    with open(output_file, "w") as f:
        f.write("\n".join(output_lines))
    console.print(f"[dim]Saved to {output_file}[/dim]")


if __name__ == "__main__":
    main()
