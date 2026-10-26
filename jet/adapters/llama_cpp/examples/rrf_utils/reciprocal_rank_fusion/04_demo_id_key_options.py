"""
Demonstrates different id_key options for identifying documents across lists.
Supports 'index', 'text', and 'id' keys.
"""

from pathlib import Path

from jet.adapters.llama_cpp.rrf_utils import reciprocal_rank_fusion
from rich.console import Console
from rich.table import Table

console = Console()

OUTPUT_DIR = Path(__file__).parent.parent / "generated" / "reciprocal_rank_fusion"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    console.print("[bold blue]RRF with Different ID Key Options[/bold blue]")
    console.print("=" * 60)

    # Example 1: Using 'index' as ID key
    console.print("\n[cyan]1. Using id_key='index'[/cyan]")

    list1_index = [
        {"index": 0, "score": 0.9, "text": "Document A"},
        {"index": 1, "score": 0.8, "text": "Document B"},
    ]

    list2_index = [
        {"index": 1, "score": 10.0, "text": "Document B"},
        {"index": 2, "score": 8.0, "text": "Document C"},
    ]

    fused_index = reciprocal_rank_fusion(
        ranked_lists=[list1_index, list2_index],
        k=60,
        weights=[1.0, 1.0],
        list_names=["list1", "list2"],
        top_n=5,
        id_key="index",
    )

    table1 = Table(
        title="Fused by Index", show_header=True, header_style="bold magenta"
    )
    table1.add_column("Rank", style="cyan")
    table1.add_column("Index", style="green")
    table1.add_column("Text", style="white")

    for r in fused_index:
        table1.add_row(str(r["rank"]), str(r["index"]), r["text"])

    console.print(table1)

    # Example 2: Using 'text' as ID key
    console.print("\n[cyan]2. Using id_key='text'[/cyan]")

    list1_text = [
        {"score": 0.9, "text": "Document A"},
        {"score": 0.8, "text": "Document B"},
    ]

    list2_text = [
        {"score": 10.0, "text": "Document B"},
        {"score": 8.0, "text": "Document C"},
    ]

    fused_text = reciprocal_rank_fusion(
        ranked_lists=[list1_text, list2_text],
        k=60,
        weights=[1.0, 1.0],
        list_names=["list1", "list2"],
        top_n=5,
        id_key="text",
    )

    table2 = Table(title="Fused by Text", show_header=True, header_style="bold magenta")
    table2.add_column("Rank", style="cyan")
    table2.add_column("Text", style="white")

    for r in fused_text:
        table2.add_row(str(r["rank"]), r["text"])

    console.print(table2)

    # Save outputs
    output_lines = [
        "ID Key Options Demo\n",
        "\n1. Fused by index:",
    ]
    for r in fused_index:
        output_lines.append(f"  #{r['rank']} idx={r['index']}: {r['text']}")

    output_lines.append("\n2. Fused by text:")
    for r in fused_text:
        output_lines.append(f"  #{r['rank']}: {r['text']}")

    output_file = OUTPUT_DIR / "04_id_key_options.txt"
    with open(output_file, "w") as f:
        f.write("\n".join(output_lines))
    console.print(f"\n[dim]Saved to {output_file}[/dim]")


if __name__ == "__main__":
    main()
