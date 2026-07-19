"""
08_demo_diversify_strategies_vector.py

Demonstrates SemHash.self_find_representative()'s `strategy` parameter,
using a custom encoder (llama.cpp server via embed_utils).

SemHash supports multiple diversification strategies (from the pyversity
library), passed via the `strategy` argument:
  - MMR   (Maximal Marginal Relevance) - the default used in prior demos
  - MSD   (Max Sum Diversification)
  - DPP   (Determinantal Point Process)
  - COVER (Coverage-based selection)
  - SSD   (Sum-of-Squared Diversification)

The index is built ONCE from precomputed embeddings, then
self_find_representative() is called once per strategy on that same index,
so any difference in the selected representatives is purely attributable
to the diversification algorithm, not to embeddings or data differences.
"""

import json
import os
import shutil
from pathlib import Path

import numpy as np
from jet.adapters.llama_cpp.embed_utils import embed
from pyversity import Strategy
from rich.console import Console
from rich.table import Table
from semhash import SemHash

console = Console()

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
console.print(f"[bold]Output dir ready:[/bold] [cyan]{OUTPUT_DIR}[/cyan]")


class LlamaCppEncoder:
    """
    Adapter that exposes our llama.cpp `embed()` helper as a SemHash-compatible
    Encoder (see semhash.utils.Encoder protocol — it only needs `.encode()`).
    """

    def encode(self, inputs, **kwargs) -> np.ndarray:
        """
        Encode a single string or a list of strings into embeddings.
        :param inputs: A string or a list of strings to encode.
        :param **kwargs: Ignored. Kept for Encoder protocol compatibility.
        :return: Embeddings as a numpy array of shape (n_inputs, embedding_dim).
        """
        texts = [inputs] if isinstance(inputs, str) else list(inputs)
        console.log(f"[LlamaCppEncoder] Encoding {len(texts)} text(s)...")
        embeddings = embed(texts, return_format="numpy", show_progress=False)
        console.log(f"[LlamaCppEncoder] Done. Output shape: {np.shape(embeddings)}")
        return embeddings


BASE_DIR = os.path.join(os.path.dirname(__file__), "mocks")
sample_data_path = os.path.join(BASE_DIR, "08_data.json")

with console.status("[bold green]Loading sample data...", spinner="dots"):
    with open(sample_data_path, "r") as f:
        sample_records = json.load(f)
console.print(f"Loaded [bold]{len(sample_records)}[/bold] sample records")

COLUMNS = ["text"]
encoder = LlamaCppEncoder()

texts = [record["text"] for record in sample_records]
console.print("[bold]Pre-computing embeddings via llama.cpp server...[/bold]")
with console.status("[bold green]Embedding sample texts...", spinner="dots"):
    doc_embeddings = embed(
        texts,
        show_progress=True,
        progress_description="Embedding sample texts",
    )
console.print(
    f"[green]Computed embeddings[/green] with shape [bold]{np.shape(doc_embeddings)}[/bold]"
)

with console.status(
    "[bold green]Building SemHash index from precomputed embeddings...", spinner="dots"
):
    semhash = SemHash.from_embeddings(
        embeddings=doc_embeddings,
        records=sample_records,
        model=encoder,
        columns=COLUMNS,
    )
console.print("[green]SemHash index built from precomputed embeddings[/green]")

SELECTION_SIZE = 6  # matches the 6 semantic clusters in 08_data.json
DIVERSITY = 0.5

STRATEGIES = [
    ("MMR", Strategy.MMR),
    ("MSD", Strategy.MSD),
    ("DPP", Strategy.DPP),
    ("COVER", Strategy.COVER),
    ("SSD", Strategy.SSD),
]

# Run each strategy against the SAME fitted index and collect results
strategy_results: dict[str, list[dict]] = {}
for name, strategy in STRATEGIES:
    with console.status(f"[bold green]Running strategy: {name}...", spinner="dots"):
        result = semhash.self_find_representative(
            selection_size=SELECTION_SIZE,
            diversity=DIVERSITY,
            strategy=strategy,
        )
    strategy_results[name] = result.selected
    console.log(f"Strategy '{name}' selected {len(result.selected)} representatives")

# Summary table: count of representatives picked per strategy (should all match SELECTION_SIZE)
summary_table = Table(
    title="Diversify Strategy Comparison — Selection Counts",
    show_header=True,
    header_style="bold magenta",
)
summary_table.add_column("Strategy", style="cyan")
summary_table.add_column("Representative count", style="green")
for name, _ in STRATEGIES:
    summary_table.add_row(name, str(len(strategy_results[name])))
console.print(summary_table)

# Side-by-side table: which text each strategy picked, one column per strategy
comparison_table = Table(
    title="Selected Representatives by Strategy",
    show_header=True,
    header_style="bold cyan",
)
comparison_table.add_column("#", style="dim")
for name, _ in STRATEGIES:
    comparison_table.add_column(name, style="white")

max_rows = max(len(v) for v in strategy_results.values())
for i in range(max_rows):
    row = [str(i + 1)]
    for name, _ in STRATEGIES:
        texts_for_strategy = strategy_results[name]
        row.append(
            texts_for_strategy[i]["text"] if i < len(texts_for_strategy) else "-"
        )
    comparison_table.add_row(*row)
console.print(comparison_table)

# Overlap check: how many representatives are shared between every pair of strategies
overlap_table = Table(
    title="Pairwise Overlap (shared representatives)",
    show_header=True,
    header_style="bold yellow",
)
overlap_table.add_column("Strategy A", style="cyan")
overlap_table.add_column("Strategy B", style="cyan")
overlap_table.add_column("Shared count", style="green")
strategy_names = [name for name, _ in STRATEGIES]
for i, name_a in enumerate(strategy_names):
    for name_b in strategy_names[i + 1 :]:
        set_a = {r["text"] for r in strategy_results[name_a]}
        set_b = {r["text"] for r in strategy_results[name_b]}
        shared = len(set_a & set_b)
        overlap_table.add_row(name_a, name_b, f"{shared}/{SELECTION_SIZE}")
console.print(overlap_table)

for name, _ in STRATEGIES:
    output_file = OUTPUT_DIR / f"representatives_{name.lower()}.json"
    with open(output_file, "w") as f:
        json.dump(strategy_results[name], f, indent=2)

console.print(
    f"[bold green]Done![/bold green] Saved per-strategy representative outputs to "
    f"[link={OUTPUT_DIR.resolve().as_uri()}]{OUTPUT_DIR}[/link]"
)
