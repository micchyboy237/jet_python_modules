"""
Demo 5: Configuration Sensitivity
---------------------------------
Goal: Observe how MinHash and Semantic thresholds impact deduplication.
Expected Outcomes:
- Lower MinHash threshold -> More aggressive lexical dedup.
- Lower Semantic threshold -> More aggressive semantic dedup.
- Helps users find the right balance for their data.
"""

import shutil
from pathlib import Path

from jet.adapters.llama_cpp.diversity_utils.fast_mmr import (
    DedupConfig,
    diversity_preserving_dedup,
)
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()
OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_TEXTS = [
    "The cat sat on the mat.",
    "The cat sat on the mat!",
    "A cat was on the mat.",
    "The dog ran in the park.",
    "A dog played outside.",
    "Birds fly in the sky.",
]


def main():
    console.print(
        Panel.fit(
            "[bold]Demo 5: Config Sensitivity[/bold]\nTuning MinHash & Semantic Thresholds",
            border_style="yellow",
        )
    )

    configs = [
        {"name": "Strict Lexical", "minhash": 0.9, "semantic": 0.95},
        {"name": "Balanced", "minhash": 0.8, "semantic": 0.9},
        {"name": "Aggressive", "minhash": 0.5, "semantic": 0.8},
    ]

    table = Table(title="Deduplication Results by Configuration")
    table.add_column("Config", style="cyan")
    table.add_column("MinHash Thresh", justify="right")
    table.add_column("Semantic Thresh", justify="right")
    table.add_column("Kept Items", justify="right")
    table.add_column("Kept Texts Preview")

    for conf in configs:
        cfg = DedupConfig(
            minhash_threshold=conf["minhash"],
            minhash_num_perm=128,
            n_clusters=2,
            semantic_sim_threshold=conf["semantic"],
        )

        kept_indices, _, _ = diversity_preserving_dedup(SAMPLE_TEXTS, cfg)
        kept_texts = [SAMPLE_TEXTS[i][:30] + "..." for i in kept_indices]

        table.add_row(
            conf["name"],
            f"{conf['minhash']:.1f}",
            f"{conf['semantic']:.2f}",
            str(len(kept_indices)),
            ", ".join(kept_texts),
        )

    console.print(table)
    console.print(
        "\n[dim]Tip: Lower MinHash threshold catches more near-duplicates. Lower Semantic threshold merges more paraphrases.[/dim]"
    )


if __name__ == "__main__":
    main()
