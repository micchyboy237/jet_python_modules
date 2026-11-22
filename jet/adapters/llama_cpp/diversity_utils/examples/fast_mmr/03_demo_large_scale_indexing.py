"""
Demo 3: Large-Scale Indexing Simulation
---------------------------------------
Goal: Demonstrate the scalability of MinHash LSH for near-duplicate detection.
Expected Outcomes:
- Fast processing of hundreds of documents.
- Effective reduction of redundant content.
- Shows how LSH avoids O(n^2) pairwise comparisons.
"""

import random
import shutil
import string
from pathlib import Path

from jet.adapters.llama_cpp.diversity_utils.fast_mmr import (
    DedupConfig,
    diversity_preserving_dedup,
)
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress

console = Console()
OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_documents(n: int) -> list[str]:
    """Generate synthetic documents with some intentional duplicates."""
    docs = []
    templates = [
        "This is a document about topic {}.",
        "Article regarding {} and its implications.",
        "Study on {} shows significant results.",
    ]
    topics = ["AI", "ML", "Data", "Code", "Web", "Cloud", "Dev", "Ops"]

    for i in range(n):
        topic = random.choice(topics)
        template = random.choice(templates)
        doc = template.format(topic)

        # Add some noise to create near-duplicates
        if random.random() < 0.3:
            doc += " " + "".join(random.choices(string.ascii_lowercase, k=5))

        docs.append(doc)

        # Intentionally add exact duplicates
        if i % 10 == 0:
            docs.append(doc)

    return docs


def main():
    console.print(
        Panel.fit(
            "[bold]Demo 3: Large-Scale Indexing[/bold]\nSimulating 500+ documents",
            border_style="blue",
        )
    )

    SAMPLE_SIZE = 500
    console.print(f"[dim]Generating {SAMPLE_SIZE} synthetic documents...[/dim]")
    texts = generate_documents(SAMPLE_SIZE)

    cfg = DedupConfig(
        minhash_threshold=0.85,
        minhash_num_perm=128,
        n_clusters=10,
        semantic_sim_threshold=0.9,
    )

    with Progress() as progress:
        task = progress.add_task("[cyan]Processing...", total=100)
        # Note: In a real scenario, we'd skip semantic dedup for speed in this demo
        # or use a smaller embedding model. Here we run the full pipeline.
        kept_indices, stats, _ = diversity_preserving_dedup(texts, cfg)
        progress.update(task, completed=100)

    console.print(f"\n[bold]Results:[/bold]")
    console.print(f"  Original: {stats['original']}")
    console.print(f"  After Exact Dedup: {stats['after_exact_dedup']}")
    console.print(f"  After Lexical Dedup: {stats['after_near_lexical_dedup']}")
    console.print(f"  After Semantic Dedup: {stats['after_semantic_dedup']}")
    console.print(
        f"  Reduction: {100 - (stats['after_semantic_dedup'] / stats['original'] * 100):.1f}%"
    )


if __name__ == "__main__":
    main()
