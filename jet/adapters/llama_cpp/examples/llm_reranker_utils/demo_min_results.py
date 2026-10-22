"""
Demo: LLM Reranker with Early Exit (min_results)

This demo illustrates how to use the `min_results` parameter to abruptly
terminate the reranking process once a sufficient number of high-quality
results have been found. This is useful for:
1. Reducing latency when only a few top results are needed.
2. Saving compute resources when dealing with large document sets.
3. Implementing "good enough" search strategies.
"""

import asyncio
import time

from jet.adapters.llama_cpp.llm_reranker_utils import arerank, rerank
from rich.console import Console

console = Console()

# Sample documents - some relevant, some not
DOCUMENTS = [
    "Python is a high-level programming language known for its readability.",
    "The capital of France is Paris.",
    "Python has a vast ecosystem for data science including Pandas and NumPy.",
    "Elephants are the largest land animals on Earth.",
    "Machine Learning in Python is facilitated by libraries like Scikit-Learn.",
    "The weather today is sunny and warm.",
    "Deep Learning frameworks like PyTorch are written in Python.",
    "Bananas are a popular fruit rich in potassium.",
    "Python's syntax allows programmers to express concepts in fewer lines of code.",
    "Water boils at 100 degrees Celsius at sea level.",
    "Data visualization in Python can be done using Matplotlib and Seaborn.",
    "The moon orbits around the Earth.",
    "Python is also used for web development with frameworks like Django.",
    "Cats are independent pets that require less maintenance than dogs.",
    "Python supports multiple programming paradigms, including procedural and object-oriented.",
]

QUERY = "Why is Python good for data science?"


def demo_sync_early_exit():
    """Demonstrates synchronous reranking with early exit."""
    console.print("\n[bold cyan]=== SYNC DEMO: Early Exit ===[/bold cyan]")

    # Without early exit (processes all batches)
    console.print("\n[yellow]1. Standard Rerank (No min_results)[/yellow]")
    start_time = time.time()
    results_full = rerank(
        query=QUERY,
        documents=DOCUMENTS,
        top_k=5,
        min_score=6.0,
        include_reasoning=False,
    )
    duration_full = time.time() - start_time
    console.print(f"   Time taken: {duration_full:.2f}s")
    console.print(f"   Results found: {len(results_full)}")

    # With early exit (stops after finding 3 results)
    console.print("\n[green]2. Early Exit Rerank (min_results=3)[/green]")
    start_time = time.time()
    results_early = rerank(
        query=QUERY,
        documents=DOCUMENTS,
        top_k=5,
        min_score=6.0,
        include_reasoning=False,
        min_results=3,
    )
    duration_early = time.time() - start_time
    console.print(f"   Time taken: {duration_early:.2f}s")
    console.print(f"   Results found: {len(results_early)}")

    if duration_full > 0:
        savings = ((duration_full - duration_early) / duration_full) * 100
        console.print(f"   [bold]Time Savings: ~{savings:.1f}%[/bold]")

    console.print("\n[bold]Top Results:[/bold]")
    for i, r in enumerate(results_early, 1):
        console.print(f"   {i}. [score:{r['score']}] {r['text'][:80]}...")


async def demo_async_early_exit():
    """Demonstrates asynchronous reranking with early exit."""
    console.print("\n[bold cyan]=== ASYNC DEMO: Early Exit ===[/bold cyan]")

    console.print("\n[green]Async Rerank with min_results=2[/green]")
    start_time = time.time()
    results = await arerank(
        query=QUERY,
        documents=DOCUMENTS,
        top_k=5,
        min_score=6.0,
        include_reasoning=True,
        min_results=2,
    )
    duration = time.time() - start_time

    console.print(f"   Time taken: {duration:.2f}s")
    console.print(f"   Results found: {len(results)}")

    console.print("\n[bold]Top Results with Reasoning:[/bold]")
    for i, r in enumerate(results, 1):
        console.print(f"   {i}. [score:{r['score']}] {r['text'][:60]}...")
        if "reason" in r:
            console.print(f"      Reason: {r['reason']}")


if __name__ == "__main__":
    console.print("[bold magenta]LLM Reranker Early Exit Demo[/bold magenta]")
    console.print(f"Query: '{QUERY}'")
    console.print(f"Total Documents: {len(DOCUMENTS)}")

    demo_sync_early_exit()
    asyncio.run(demo_async_early_exit())

    console.print("\n[bold green]Demo Complete[/bold green]")
