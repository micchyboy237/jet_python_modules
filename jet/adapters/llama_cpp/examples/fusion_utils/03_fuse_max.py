"""Max Score Fusion Demo.

Real-world use case: Safety-critical or compliance retrieval where missing
a relevant document is unacceptable. If ANY signal strongly matches a
document, it should surface regardless of other signals. Common in legal
review, medical literature search, and regulatory compliance pipelines.
"""

import json
import shutil
from pathlib import Path

import numpy as np
from jet.adapters.llama_cpp.examples.fusion_utils._helpers import scores_in_doc_order
from jet.adapters.llama_cpp.fusion_utils import fuse_max
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

SETTINGS = {
    "strategy": "max_score",
    "normalize": True,
    "description": "Conservative fusion: any strong signal surfaces the document",
}

QUERY = "What do pandas eat?"

DOCUMENTS = [
    "The giant panda is a bear species endemic to China.",
    "Python is a high-level programming language.",
    "Bears are carnivoran mammals of the family Ursidae.",
    "Machine learning is a subset of artificial intelligence.",
    "Pandas eat bamboo and live in mountainous regions.",
]
n_docs = len(DOCUMENTS)

logger.info(
    f"Computing raw signals for query='{QUERY}' (fuse_max normalizes internally)"
)
vector_results = vector_search(QUERY, DOCUMENTS)
keyword_results = rerank(QUERY, DOCUMENTS, method="bm25", normalize_scores=False)
reranker_results = rerank(QUERY, DOCUMENTS, method="model", normalize_scores=False)

INPUTS = {
    "query": QUERY,
    "documents": DOCUMENTS,
    "signal_scores": {
        "embedding": scores_in_doc_order(vector_results, n_docs, "score"),
        "keyword": scores_in_doc_order(keyword_results, n_docs, "raw_score"),
        "reranker": scores_in_doc_order(reranker_results, n_docs, "raw_score"),
    },
}
logger.info(f"raw signal_scores: {INPUTS['signal_scores']}")

console.print(Panel("🛡️ Max Score Fusion Demo", style="bold cyan"))

console.print("\n[bold]Signal Scores (raw, will be minmax-normalized):[/bold]")
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

fused = fuse_max(
    signal_scores=INPUTS["signal_scores"],
    normalize=SETTINGS["normalize"],
)

sorted_indices = np.argsort(fused)[::-1]

console.print("\n[bold]Fused Results (max aggregation):[/bold]")
result_table = Table(show_header=True, header_style="bold green")
result_table.add_column("Rank", style="dim", width=6)
result_table.add_column("Index", justify="center", width=6)
result_table.add_column("Max Score", justify="right", width=10)
result_table.add_column("Document Text")
for rank, idx in enumerate(sorted_indices, 1):
    result_table.add_row(
        str(rank), str(idx), f"{fused[idx]:.4f}", INPUTS["documents"][idx][:60]
    )
console.print(result_table)

# Show which signal won for each document
console.print("\n[bold]Winning Signal Per Document:[/bold]")
win_table = Table(show_header=True, header_style="bold yellow")
win_table.add_column("Doc Index", justify="center", width=10)
win_table.add_column("Max Score", justify="right", width=10)
win_table.add_column("Winning Signal")
for idx in sorted_indices:
    best_sig = max(
        INPUTS["signal_scores"].keys(),
        key=lambda s: INPUTS["signal_scores"][s][idx],
    )
    win_table.add_row(str(idx), f"{fused[idx]:.4f}", best_sig)
console.print(win_table)

outputs = {
    "fused_scores": fused.tolist(),
    "final_ranking": sorted_indices.tolist(),
    "winning_signals": {
        str(idx): max(
            INPUTS["signal_scores"].keys(),
            key=lambda s: INPUTS["signal_scores"][s][idx],
        )
        for idx in range(len(INPUTS["documents"]))
    },
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
