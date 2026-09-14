"""Reciprocal Rank Fusion (RRF) Demo.

Real-world use case: Combining BM25 keyword search results with vector
semantic search results when their raw scores are on incomparable scales.
RRF uses only rank positions, making it robust to score distribution
differences without any normalization or weight tuning.
"""

import json
import shutil
from pathlib import Path

from jet.adapters.llama_cpp.fusion_utils import fuse_rrf
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Simulated rankings from two different retrieval systems
# Each list contains document indices ordered by relevance (best first)
SETTINGS = {
    "strategy": "rrf",
    "k": 60,
    "description": "Combining BM25 and vector search rankings via RRF",
}

INPUTS = {
    "bm25_ranking": [2, 0, 4, 1, 3],
    "vector_ranking": [0, 1, 2, 3, 4],
    "documents": [
        "Python is a high-level programming language.",
        "Machine learning is a subset of artificial intelligence.",
        "The giant panda is a bear species endemic to China.",
        "JavaScript is commonly used for web development.",
        "Pandas eat bamboo and live in mountainous regions.",
    ],
}

console.print(Panel("🔀 RRF Fusion Demo", style="bold cyan"))

console.print("\n[bold]Input Rankings:[/bold]")
table = Table(show_header=True, header_style="bold magenta")
table.add_column("Rank", style="dim", width=6)
table.add_column("BM25 Doc Index", justify="center")
table.add_column("Vector Doc Index", justify="center")
for i in range(len(INPUTS["bm25_ranking"])):
    table.add_row(
        str(i + 1),
        str(INPUTS["bm25_ranking"][i]),
        str(INPUTS["vector_ranking"][i]),
    )
console.print(table)

rankings = [INPUTS["bm25_ranking"], INPUTS["vector_ranking"]]
fused_scores = fuse_rrf(rankings=rankings, k=SETTINGS["k"])

# Build sorted results
indexed_scores = list(enumerate(fused_scores))
indexed_scores.sort(key=lambda x: x[1], reverse=True)

console.print(f"\n[bold]Fused Results (k={SETTINGS['k']}):[/bold]")
result_table = Table(show_header=True, header_style="bold green")
result_table.add_column("Final Rank", style="dim", width=10)
result_table.add_column("Doc Index", justify="center", width=10)
result_table.add_column("RRF Score", justify="right", width=12)
result_table.add_column("Document Text")
for rank, (doc_idx, score) in enumerate(indexed_scores, 1):
    result_table.add_row(
        str(rank),
        str(doc_idx),
        f"{score:.6f}",
        INPUTS["documents"][doc_idx][:60],
    )
console.print(result_table)

outputs = {
    "fused_scores": fused_scores.tolist(),
    "final_ranking": [idx for idx, _ in indexed_scores],
    "results": [
        {"rank": r, "index": idx, "score": float(s), "text": INPUTS["documents"][idx]}
        for r, (idx, s) in enumerate(indexed_scores, 1)
    ],
}

with open(OUTPUT_DIR / "settings.json", "w") as f:
    json.dump(SETTINGS, f, indent=2)
with open(OUTPUT_DIR / "inputs.json", "w") as f:
    json.dump(INPUTS, f, indent=2)
with open(OUTPUT_DIR / "outputs.json", "w") as f:
    json.dump(outputs, f, indent=2)

console.print(f"\n[dim]Outputs saved to {OUTPUT_DIR}[/dim]")
