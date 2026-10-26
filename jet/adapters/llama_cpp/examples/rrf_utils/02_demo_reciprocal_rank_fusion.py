import shutil
from pathlib import Path

from jet.adapters.llama_cpp.rrf_utils import reciprocal_rank_fusion
from rich.console import Console
from rich.tree import Tree

console = Console()

# --- Configuration ---
OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DOCS = [
    "The giant panda is a bear species endemic to China.",
    "Python is a high-level programming language.",
    "Bears are carnivoran mammals of the family Ursidae.",
    "Machine learning is a subset of artificial intelligence.",
    "Pandas eat bamboo and live in mountainous regions.",
]


def run_manual_fusion():
    """Example 1: Fusing two manually created ranked lists"""
    console.print("[bold blue]Running Manual RRF Fusion...[/bold blue]")

    # Simulate results from two different systems
    dense_results = [
        {"index": 0, "text": DOCS[0], "score": 0.9},
        {"index": 4, "text": DOCS[4], "score": 0.8},
        {"index": 1, "text": DOCS[1], "score": 0.7},
    ]

    bm25_results = [
        {"index": 0, "text": DOCS[0], "score": 10.5},
        {"index": 2, "text": DOCS[2], "score": 8.2},
        {"index": 4, "text": DOCS[4], "score": 6.1},
    ]

    fused = reciprocal_rank_fusion(
        ranked_lists=[dense_results, bm25_results],
        k=60,
        weights=[1.0, 1.0],
        list_names=["Dense", "BM25"],
        top_n=5,
        id_key="index",
    )

    # Save Output
    output_file = OUTPUT_DIR / "manual_fusion_results.txt"
    with open(output_file, "w") as f:
        f.write("Manual RRF Fusion Results\n\n")
        for r in fused:
            f.write(f"Rank {r['rank']}: {r['text']}\n")
            f.write(f"  Sources: {', '.join(r['sources'])}\n")
            f.write(f"  Original Ranks: {r['original_ranks']}\n\n")

    console.print(f"[green]Saved results to:[/green] {output_file}")
    return fused


def run_weighted_fusion():
    """Example 2: Fusing with custom weights"""
    console.print("\n[bold blue]Running Weighted RRF Fusion...[/bold blue]")

    # Same lists as above, but we trust Dense more
    dense_results = [
        {"index": 0, "text": DOCS[0], "score": 0.9},
        {"index": 1, "text": DOCS[1], "score": 0.8},
    ]

    bm25_results = [
        {"index": 0, "text": DOCS[0], "score": 10.5},
        {"index": 2, "text": DOCS[2], "score": 8.2},
    ]

    fused = reciprocal_rank_fusion(
        ranked_lists=[dense_results, bm25_results],
        k=60,
        weights=[2.0, 0.5],  # Dense is 4x more influential than BM25 here
        list_names=["Dense (High Trust)", "BM25 (Low Trust)"],
        top_n=5,
        id_key="index",
    )

    # Save Output
    output_file = OUTPUT_DIR / "weighted_fusion_results.txt"
    with open(output_file, "w") as f:
        f.write("Weighted RRF Fusion Results (Dense=2.0, BM25=0.5)\n\n")
        for r in fused:
            f.write(f"Rank {r['rank']}: {r['text']}\n")
            f.write(f"  RRF Score: {r['score']:.5f}\n\n")

    console.print(f"[green]Saved results to:[/green] {output_file}")
    return fused


def show_file_tree():
    """Display the generated files as a tree with links."""
    tree = Tree("📂 Generated Outputs")
    for file_path in sorted(OUTPUT_DIR.rglob("*")):
        if file_path.is_file():
            relative_path = file_path.relative_to(OUTPUT_DIR.parent.parent)
            # Create a resource link for the terminal
            link_text = f"[link=file://{file_path}]{file_path.name}[/link]"
            tree.add(f"{link_text} ({file_path.stat().st_size} bytes)")

    console.print("\n[bold yellow]Generated Files:[/bold yellow]")
    console.print(tree)


if __name__ == "__main__":
    console.print(f"[dim]Output Directory: {OUTPUT_DIR}[/dim]\n")

    run_manual_fusion()
    run_weighted_fusion()

    show_file_tree()
