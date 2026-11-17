"""
Demonstrates basic diversity-aware result selection using MMR.
Selects a diverse subset from a pre-ranked list of candidates.
Uses pre-computed embeddings instead of a fetcher function.
"""

import json
import shutil
from pathlib import Path

from jet.adapters.llama_cpp.diversity_utils import diversify_results
from jet.adapters.llama_cpp.embed_utils import embed
from rich.console import Console

console = Console()

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

documents = [
    "Python is a popular programming language for web development.",
    "The Python snake is a large constrictor found in Africa and Asia.",
    "Python software foundation supports the development of Python.",
    "Monty Python was a British surreal comedy group.",
    "Learning Python syntax is easy for beginners.",
    "Python boots are a type of footwear named after the snake.",
    "Data science libraries like Pandas are written in Python.",
    "The Monty Python film 'Life of Brian' is a classic.",
]

results = [
    {"id": f"doc_{i}", "document": doc, "score": 1.0 - (i * 0.05)}
    for i, doc in enumerate(documents)
]

# Pre-compute embeddings for all documents
console.print("[dim]Generating embeddings for candidate pool...[/dim]")
all_embeddings = embed([r["document"] for r in results], show_progress=False)

console.print("[bold blue]=== Basic Diversity Demo ===[/bold blue]")
console.print(f"Candidate pool size: {len(results)}")
console.print(f"Target top_k: 4")
console.print(f"Diversity level: 0.8 (High diversity)")

# Pass pre-computed embeddings directly
diversified = diversify_results(
    results=results, top_k=4, diversity=0.8, embeddings=all_embeddings
)

console.print("\n[bold green]Selected Diverse Results:[/bold green]")
for i, item in enumerate(diversified):
    console.print(
        f"{i + 1}. [cyan]{item['document']}[/cyan] (Score: {item['score']:.2f})"
    )

output_file = OUTPUT_DIR / "diversified_results.json"
output_file.write_text(json.dumps(diversified, indent=2))

console.print(f"\n[bold]Saved files:[/]")
for f in sorted(OUTPUT_DIR.iterdir()):
    console.print(f"  📄 [link=file://{f}]{f.name}[/link]")
