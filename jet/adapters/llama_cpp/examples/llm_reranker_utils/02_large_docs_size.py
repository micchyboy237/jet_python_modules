"""Demo 2: Multi-Batch Processing (With Splitting)
This example demonstrates the greedy bin-packing algorithm. By setting a
specific token limit, we force the 5 documents into 3 distinct batches,
allowing for relative comparison within each group.
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
    # Set to 200 to force splitting into ~3 batches (Budget ~93 tokens/batch)
    reranker = LLMReranker(max_tokens=200, model=LLM_MODEL)
    query = "Explain the concept of machine learning"
    documents = [
        "Machine learning is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. It focuses on the development of computer programs that can access data and use it to learn for themselves.",
        "Supervised learning involves training a model on a labeled dataset. The algorithm learns to map inputs to outputs based on example input-output pairs. Common tasks include classification and regression.",
        "Unsupervised learning deals with unlabeled data. The system tries to learn the natural structure present within the input data. Clustering and dimensionality reduction are common unsupervised tasks.",
        "Reinforcement learning is an area of machine learning concerned with how software agents ought to take actions in an environment so as to maximize some notion of cumulative reward.",
        "Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised.",
    ]

    # Calculate tokens
    doc_tokens = count_tokens(documents, model=LLM_MODEL, prevent_total=True)
    total_tokens = sum(doc_tokens)

    # Log to console
    console.print(
        Panel.fit(
            "[bold red]DEMO 2: Multi-Batch Processing (3 Batches)[/bold red]",
            border_style="red",
        )
    )
    console.print(f"[cyan]Query:[/cyan] {query}")
    console.print(f"[cyan]Documents:[/cyan] {len(documents)}")
    console.print(f"[cyan]Doc Tokens:[/cyan] {doc_tokens}")
    console.print(f"[cyan]Total Tokens:[/cyan] {total_tokens}")
    console.print(f"[cyan]Max Tokens per Batch:[/cyan] 200")
    console.print()
    console.print(
        "[italic yellow]Note: Watch the logs to see the system create exactly 3 batches.[/italic yellow]"
    )
    console.print()

    results = reranker.rerank(
        query=query,
        documents=documents,
        top_k=3,
        min_score=5.0,
        include_reasoning=True,
    )

    table = Table(
        title="Globally Sorted Results", show_header=True, header_style="bold magenta"
    )
    table.add_column("Rank", style="dim", width=4)
    table.add_column("Global Doc Index", justify="center")
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
                f.write(f"{i}. Global Doc[{res['index']}] - Score: {res['score']}\n")
                f.write(f"   Reason: {res.get('reason', 'N/A')}\n")

        # Save metadata
        metadata = {
            "query": query,
            "num_documents": len(documents),
            "doc_tokens": doc_tokens,
            "total_tokens": total_tokens,
            "max_tokens_limit": 200,
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
