"""
Demo 2: Semantic vs. Lexical Dedup
----------------------------------
Goal: Contrast MinHash (lexical) with Embedding-based (semantic) dedup.
Expected Outcomes:
- MinHash keeps paraphrases because they share few words.
- Semantic dedup removes paraphrases because they have high cosine similarity.
- Demonstrates why both layers are necessary for robust cleaning.
"""

import shutil
from pathlib import Path

from jet.adapters.llama_cpp.diversity_utils.fast_mmr import (
    DedupConfig,
    embed_texts,
    exact_dedup,
    near_lexical_dedup,
    semantic_dedup,
)
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()
OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Paraphrases: Lexically different, semantically identical
SAMPLE_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "A fast auburn canine leaps above the indolent hound.",  # Paraphrase
    "Python is a great language for data science.",
    "For data science tasks, Python is an excellent choice.",  # Paraphrase
    "The weather is sunny today.",
]


def main():
    console.print(
        Panel.fit(
            "[bold]Demo 2: Semantic vs. Lexical Dedup[/bold]\nParaphrase Detection",
            border_style="magenta",
        )
    )

    cfg = DedupConfig(
        minhash_threshold=0.5,  # Lower threshold to be less aggressive lexically
        minhash_num_perm=128,
        n_clusters=2,
        semantic_sim_threshold=0.85,
    )

    console.print("[dim]Input Texts:[/dim]")
    for i, t in enumerate(SAMPLE_TEXTS):
        console.print(f"  [{i}] {t}")

    # Step 1: Exact Dedup
    idx_exact = exact_dedup(SAMPLE_TEXTS)
    console.print(f"\nAfter Exact Dedup: {len(idx_exact)} items")

    # Step 2: Near-Lexical Dedup (MinHash)
    idx_lexical = near_lexical_dedup(SAMPLE_TEXTS, idx_exact, cfg)
    console.print(f"After Near-Lexical Dedup (MinHash): {len(idx_lexical)} items")
    console.print(
        "[dim]Note: Paraphrases often survive MinHash because they share few tokens.[/dim]"
    )

    # Step 3: Semantic Dedup (Embeddings)
    embeddings = embed_texts(SAMPLE_TEXTS, idx_lexical, cfg)
    idx_semantic, _ = semantic_dedup(embeddings, idx_lexical, cfg)
    console.print(f"After Semantic Dedup (Embeddings): {len(idx_semantic)} items")

    table = Table(title="Final Unique Semantic Concepts")
    table.add_column("Index", style="cyan")
    table.add_column("Text")
    for idx in idx_semantic:
        table.add_row(str(idx), SAMPLE_TEXTS[idx])
    console.print(table)


if __name__ == "__main__":
    main()
