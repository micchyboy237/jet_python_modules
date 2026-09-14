"""Weighted Sum Fusion Demo.

Real-world use case: Ensemble search where you've tuned signal weights
through A/B testing (e.g., 40% embedding, 30% keyword, 30% reranker).
Best when signals are pre-normalized to comparable scales and you want
fine-grained control over each signal's contribution.
"""

import json
import shutil
from pathlib import Path

import numpy as np
from jet.adapters.llama_cpp.examples.fusion_utils._helpers import scores_in_doc_order
from jet.adapters.llama_cpp.fusion_utils import fuse_weighted_sum
from jet.adapters.llama_cpp.rerank_utils import rerank
from jet.adapters.llama_cpp.vector_utils import vector_search
from jet.logger import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- SETTINGS: change normalize to True (real signals aren't pre-scaled) ---
SETTINGS = {
    "strategy": "weighted_sum",
    "weights": {"embedding": 0.4, "keyword": 0.3, "reranker": 0.3},
    "normalize": True,
    "description": "Live signals (embedding/BM25/reranker) combined with tuned weights",
}

# --- replace the whole INPUTS block with: ---
QUERY = "What do pandas eat?"

DOCUMENTS = [
    "The giant panda is a bear species endemic to China.",
    "Python is a high-level programming language.",
    "Bears are carnivoran mammals of the family Ursidae.",
    "Machine learning is a subset of artificial intelligence.",
    "Pandas eat bamboo and live in mountainous regions.",
]
n_docs = len(DOCUMENTS)

logger.info(f"Computing embedding similarity for query='{QUERY}'")
vector_results = vector_search(QUERY, DOCUMENTS)

logger.info("Computing BM25 keyword scores")
keyword_results = rerank(QUERY, DOCUMENTS, method="bm25", normalize_scores=True)

logger.info("Computing cross-encoder reranker scores")
reranker_results = rerank(QUERY, DOCUMENTS, method="model", normalize_scores=True)

INPUTS = {
    "query": QUERY,
    "documents": DOCUMENTS,
    "signal_scores": {
        "embedding": scores_in_doc_order(vector_results, n_docs, "score"),
        "keyword": scores_in_doc_order(keyword_results, n_docs, "score"),
        "reranker": scores_in_doc_order(reranker_results, n_docs, "score"),
    },
}
logger.info(f"signal_scores populated: {INPUTS['signal_scores']}")

console.print(Panel("⚖️ Weighted Sum Fusion Demo", style="bold cyan"))

console.print("\n[bold]Signal Scores (pre-normalized):[/bold]")
table = Table(show_header=True, header_style="bold magenta")
table.add_column("Doc", style="dim", width=4)
for sig in INPUTS["signal_scores"]:
    table.add_column(sig.capitalize(), justify="right", width=12)
for i in range(len(INPUTS["documents"])):
    row = [str(i)]
    for sig in INPUTS["signal_scores"]:
        row.append(f"{INPUTS['signal_scores'][sig][i]:.3f}")
    table.add_row(*row)
console.print(table)

console.print(f"\n[bold]Weights:[/bold] {SETTINGS['weights']}")

fused = fuse_weighted_sum(
    signal_scores=INPUTS["signal_scores"],
    weights=SETTINGS["weights"],
    normalize=SETTINGS["normalize"],
)

sorted_indices = np.argsort(fused)[::-1]

console.print("\n[bold]Fused Results:[/bold]")
result_table = Table(show_header=True, header_style="bold green")
result_table.add_column("Rank", style="dim", width=6)
result_table.add_column("Index", justify="center", width=6)
result_table.add_column("Score", justify="right", width=10)
result_table.add_column("Document Text")
for rank, idx in enumerate(sorted_indices, 1):
    result_table.add_row(
        str(rank), str(idx), f"{fused[idx]:.4f}", INPUTS["documents"][idx][:60]
    )
console.print(result_table)

outputs = {
    "fused_scores": fused.tolist(),
    "final_ranking": sorted_indices.tolist(),
    "results": [
        {
            "rank": r,
            "index": int(idx),
            "score": float(fused[idx]),
            "text": INPUTS["documents"][idx],
        }
        for r, idx in enumerate(sorted_indices, 1)
    ],
}

with open(OUTPUT_DIR / "settings.json", "w") as f:
    json.dump(SETTINGS, f, indent=2)
with open(OUTPUT_DIR / "inputs.json", "w") as f:
    json.dump(INPUTS, f, indent=2)
with open(OUTPUT_DIR / "outputs.json", "w") as f:
    json.dump(outputs, f, indent=2)

console.print(f"\n[dim]Outputs saved to {OUTPUT_DIR}[/dim]")
