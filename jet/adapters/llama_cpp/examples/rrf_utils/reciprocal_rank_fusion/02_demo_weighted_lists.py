"""
RRF fusion with custom weights for each ranked list.
Use when some retrieval methods are more trustworthy than others.
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
    console.print("[bold blue]Weighted RRF Fusion[/bold blue]")
    console.print("dense weight=2.0, bm25 weight=0.5")
    console.print("=" * 60)

    dense_results = [
        {"index": 0, "score": 0.92, "text": DOCUMENTS[0]},
        {"index": 1, "score": 0.80, "text": DOCUMENTS[1]},
        {"index": 4, "score": 0.75, "text": DOCUMENTS[4]},
    ]

    bm25_results = [
        {"index": 0, "score": 12.4, "text": DOCUMENTS[0]},
        {"index": 2, "score": 8.2, "text": DOCUMENTS[2]},
        {"index": 4, "score": 6.1, "text": DOCUMENTS[4]},
    ]

    fused = reciprocal_rank_fusion(
        ranked_lists=[dense_results, bm25_results],
        k=60,
        weights=[2.0, 0.5],  # Dense is 4x more important than BM25
        list_names=["dense (high trust)", "bm25 (low trust)"],
        top_n=5,
        id_key="index",
    )

    table = Table(
        title="Weighted Fused Results", show_header=True, header_style="bold magenta"
    )
    table.add_column("Rank", style="cyan")
    table.add_column("RRF Score", style="yellow")
    table.add_column("Sources", style="blue")
    table.add_column("Text", style="white")

    output_lines = ["Weighted RRF Fusion (dense=2.0, bm25=0.5)\n"]

    for r in fused:
        table.add_row(
            str(r["rank"]),
            f"{r['score']:.5f}",
            ", ".join(r["sources"]),
            r["text"][:60],
        )
        output_lines.append(f"#{r['rank']} (RRF={r['score']:.5f}): {r['text']}")

    console.print(table)

    output_file = OUTPUT_DIR / "02_weighted_lists.txt"
    with open(output_file, "w") as f:
        f.write("\n".join(output_lines))
    console.print(f"[dim]Saved to {output_file}[/dim]")


if __name__ == "__main__":
    main()
