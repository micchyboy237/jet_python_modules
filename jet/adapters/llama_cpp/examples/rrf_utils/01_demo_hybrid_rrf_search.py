import shutil
from pathlib import Path

from jet.adapters.llama_cpp.rrf_utils import hybrid_rrf_search
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
    "The red panda is a small mammal native to the eastern Himalayas.",
    "Bamboo is the primary food source for giant pandas.",
]


def run_basic_hybrid_search():
    """Example 1: Basic Hybrid Search (Dense + BM25)"""
    console.print("[bold blue]Running Basic Hybrid RRF Search...[/bold blue]")

    query = "What is a giant panda?"
    results = hybrid_rrf_search(
        query=query,
        documents=DOCS,
        dense_top_n=5,
        sparse_top_n=5,
        final_top_n=3,
        use_rerank=False,
    )

    # Save Output
    output_file = OUTPUT_DIR / "basic_hybrid_results.txt"
    with open(output_file, "w") as f:
        f.write(f"Query: {query}\n\n")
        for r in results:
            f.write(f"Rank {r['rank']}: {r['text']} (Score: {r['score']:.4f})\n")

    console.print(f"[green]Saved results to:[/green] {output_file}")
    return results


def run_weighted_hybrid_search():
    """Example 2: Weighted Hybrid Search (Prioritizing Dense)"""
    console.print("\n[bold blue]Running Weighted Hybrid RRF Search...[/bold blue]")

    query = "Programming languages for AI"
    results = hybrid_rrf_search(
        query=query,
        documents=DOCS,
        dense_top_n=5,
        sparse_top_n=5,
        final_top_n=3,
        dense_weight=2.0,  # Trust semantic search more
        sparse_weight=0.5,  # Trust keyword search less
        use_rerank=False,
    )

    # Save Output
    output_file = OUTPUT_DIR / "weighted_hybrid_results.txt"
    with open(output_file, "w") as f:
        f.write(f"Query: {query}\nWeights: Dense=2.0, Sparse=0.5\n\n")
        for r in results:
            f.write(f"Rank {r['rank']}: {r['text']} (Score: {r['score']:.4f})\n")

    console.print(f"[green]Saved results to:[/green] {output_file}")
    return results


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

    run_basic_hybrid_search()
    run_weighted_hybrid_search()

    show_file_tree()
