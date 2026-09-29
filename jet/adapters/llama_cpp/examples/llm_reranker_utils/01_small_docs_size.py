"""Demo 1: Single Batch Processing (No Splitting)
This example demonstrates a scenario where the total token count of all
documents is well within the max_tokens limit. The reranker processes
all documents in a single batch.
"""

import json
import shutil
from pathlib import Path

from jet.adapters.llama_cpp.config import LLM_MODEL
from jet.adapters.llama_cpp.llm_reranker_utils import LLMReranker
from jet.adapters.llama_cpp.token_utils import count_tokens
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Setup Output Directory
OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

console = Console()


def main():
    reranker = LLMReranker(max_tokens=1000, model=LLM_MODEL)
    query = "What are the key features of Python?"
    documents = [
        "Python is known for its readable syntax and ease of use.",
        "It has a vast standard library often described as 'batteries included'.",
        "Python supports multiple programming paradigms including OOP and functional.",
        "The language uses dynamic typing and automatic memory management.",
    ]

    # Calculate tokens
    doc_tokens = count_tokens(documents, model=LLM_MODEL, prevent_total=True)
    total_tokens = sum(doc_tokens)

    # Log to console
    console.print(
        Panel.fit(
            "[bold blue]DEMO 1: Single Batch Processing[/bold blue]",
            border_style="blue",
        )
    )
    console.print(f"[cyan]Query:[/cyan] {query}")
    console.print(f"[cyan]Documents:[/cyan] {len(documents)}")
    console.print(f"[cyan]Doc Tokens:[/cyan] {doc_tokens}")
    console.print(f"[cyan]Total Tokens:[/cyan] {total_tokens}")
    console.print(f"[cyan]Max Tokens per Batch:[/cyan] 1000")
    console.print()

    results = reranker.rerank(
        query=query,
        documents=documents,
        top_k=3,
        min_score=5.0,
        include_reasoning=True,
    )

    table = Table(
        title="Ranking Results", show_header=True, header_style="bold magenta"
    )
    table.add_column("Rank", style="dim", width=4)
    table.add_column("Doc Index", justify="center")
    table.add_column("Score", justify="right")
    table.add_column("Reason")

    if results:
        for i, res in enumerate(results, 1):
            table.add_row(
                str(i),
                str(res["index"]),
                f"{res['score']:.1f}",
                res.get("reason", "N/A"),
            )
        console.print(table)

        # Save results and metadata
        output_file = OUTPUT_DIR / "results.txt"
        with open(output_file, "w") as f:
            f.write(f"Query: {query}\n\n")
            for i, res in enumerate(results, 1):
                f.write(f"{i}. Doc[{res['index']}] - Score: {res['score']}\n")
                f.write(f"   Reason: {res.get('reason', 'N/A')}\n")

        # Save metadata
        metadata = {
            "query": query,
            "num_documents": len(documents),
            "doc_tokens": doc_tokens,
            "total_tokens": total_tokens,
            "max_tokens_limit": 1000,
            "num_results": len(results),
        }
        metadata_file = OUTPUT_DIR / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        console.print(f"\n[green]Results saved to:[/green] {output_file.name}")
        console.print(f"[green]Metadata saved to:[/green] {metadata_file.name}")
    else:
        console.print("[yellow]No documents met the threshold.[/yellow]")

    console.print()


if __name__ == "__main__":
    main()

    # Display resource link
    print(f"resource://{Path(__file__).stem}")
