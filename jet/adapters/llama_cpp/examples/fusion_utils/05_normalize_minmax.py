"""Min-Max Normalization Demo.

Real-world use case: Pre-processing step before weighted sum fusion when
signals have different scales (e.g., BM25 scores range 0–25 while cosine
similarity ranges 0–1). Ensures no single signal dominates due to scale
alone. Also useful for displaying heterogeneous scores in unified UIs.
"""

import json
import shutil
from pathlib import Path

from jet.adapters.llama_cpp.examples.fusion_utils._helpers import scores_in_doc_order
from jet.adapters.llama_cpp.fusion_utils import normalize_minmax
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
    "normalizer": "minmax",
    "description": "Scaling heterogeneous signals to [0, 1] for fair comparison",
}

QUERY = "What do pandas eat?"

DOCUMENTS = [
    "JavaScript is commonly used for web development.",
    "Bears are carnivoran mammals of the family Ursidae.",
    "Machine learning is a subset of artificial intelligence.",
    "The giant panda is a bear species endemic to China.",
    "Pandas eat bamboo and live in mountainous regions.",
]
n_docs = len(DOCUMENTS)

logger.info(f"Fetching raw signals (different natural scales) for query='{QUERY}'")
bm25_results = rerank(QUERY, DOCUMENTS, method="bm25", normalize_scores=False)
vector_results = vector_search(QUERY, DOCUMENTS)
reranker_results = rerank(QUERY, DOCUMENTS, method="model", normalize_scores=False)

INPUTS = {
    "signals": {
        "bm25_raw": scores_in_doc_order(bm25_results, n_docs, "raw_score"),
        "cosine_similarity": scores_in_doc_order(vector_results, n_docs, "score"),
        "reranker_logits": scores_in_doc_order(reranker_results, n_docs, "raw_score"),
    },
    "documents": DOCUMENTS,
}
logger.info(f"raw signals: {INPUTS['signals']}")

console.print(Panel("📏 Min-Max Normalization Demo", style="bold cyan"))

console.print("\n[bold]Raw Signals (different scales):[/bold]")
raw_table = Table(show_header=True, header_style="bold magenta")
raw_table.add_column("Doc", style="dim", width=4)
for sig in INPUTS["signals"]:
    raw_table.add_column(sig, justify="right", width=18)
for i in range(len(INPUTS["documents"])):
    row = [str(i)]
    for sig in INPUTS["signals"]:
        row.append(f"{INPUTS['signals'][sig][i]:.2f}")
    raw_table.add_row(*row)
console.print(raw_table)

normalized = {}
for sig_name, scores in INPUTS["signals"].items():
    normalized[sig_name] = normalize_minmax(scores).tolist()

console.print("\n[bold]After Min-Max Normalization [0, 1]:[/bold]")
norm_table = Table(show_header=True, header_style="bold green")
norm_table.add_column("Doc", style="dim", width=4)
for sig in INPUTS["signals"]:
    norm_table.add_column(sig, justify="right", width=18)
for i in range(len(INPUTS["documents"])):
    row = [str(i)]
    for sig in INPUTS["signals"]:
        row.append(f"{normalized[sig][i]:.4f}")
    norm_table.add_row(*row)
console.print(norm_table)

console.print(
    "\n[yellow]💡 All signals now contribute equally to weighted sum fusion[/yellow]"
)
console.print(
    "[yellow]   ⚠️  Sensitive to outliers — one extreme value compresses others[/yellow]"
)

outputs = {
    "normalized_signals": normalized,
    "original_ranges": {
        sig: {"min": min(vals), "max": max(vals)}
        for sig, vals in INPUTS["signals"].items()
    },
}

with open(OUTPUT_DIR / "settings.json", "w") as f:
    json.dump(SETTINGS, f, indent=2)
with open(OUTPUT_DIR / "inputs.json", "w") as f:
    json.dump(INPUTS, f, indent=2)
with open(OUTPUT_DIR / "outputs.json", "w") as f:
    json.dump(outputs, f, indent=2)

console.print(f"\n[dim]Outputs saved to {OUTPUT_DIR}[/dim]")
