"""
04_demo_filter_outliers_vector.py

Demonstrates SemHash.self_filter_outliers() built from *pre-computed*
embeddings (via our local llama.cpp embedding server) instead of the
default SemHash model.

Important distinction from 02_demo_find_repr_vector.py:
- self_find_representative() re-encodes shortlisted candidates internally
  (via _diversify -> model.encode) during MMR reranking, so the passed-in
  model matters at query time too.
- self_filter_outliers() does NOT re-encode anything. It only ranks using
  the embeddings already stored in the index (index.vectors). So the
  `model` argument to from_embeddings() here is only used to satisfy the
  Encoder protocol required by SemHash's type signature — it is never
  actually invoked after the index is built.

This demo mainly shows: how to build a SemHash index from embeddings you
already computed yourself (e.g. via a local server), instead of letting
SemHash download/call the default HuggingFace model.

Uses the same mock dataset as 03_demo_filter_outliers.py so you can compare
whether the local llama.cpp model flags the same outliers as the default
SemHash model.
"""

import json
import os
import shutil
from pathlib import Path

import numpy as np
from jet.adapters.llama_cpp.embed_utils import embed
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

    Note: for self_filter_outliers(), this encode() method is NOT called
    after index construction — outlier ranking uses the index's cached
    vectors directly. This class exists only to satisfy SemHash's required
    `model` parameter in from_embeddings().
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
sample_data_path = os.path.join(BASE_DIR, "04_data.json")

with console.status("[bold green]Loading sample data...", spinner="dots"):
    with open(sample_data_path, "r") as f:
        sample_records = json.load(f)
console.print(f"Loaded [bold]{len(sample_records)}[/bold] sample records")

texts = [record["text"] for record in sample_records]
console.log(f"Extracted {len(texts)} texts for embedding")

encoder = LlamaCppEncoder()

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
        columns=["text"],
    )
console.print("[green]SemHash index built from precomputed embeddings[/green]")

OUTLIER_PERCENTAGE = 0.2  # bottom 20% least-similar records flagged as outliers

with console.status("[bold green]Filtering outliers...", spinner="dots"):
    result = semhash.self_filter_outliers(outlier_percentage=OUTLIER_PERCENTAGE)

inlier_count = len(result.selected)
outlier_count = len(result.filtered)

table = Table(
    title="Outlier Filtering Results (from precomputed embeddings)",
    show_header=True,
    header_style="bold magenta",
)
table.add_column("Metric", style="cyan")
table.add_column("Value", style="green")
table.add_row("Input records", str(len(sample_records)))
table.add_row("Outlier percentage", f"{OUTLIER_PERCENTAGE:.0%}")
table.add_row("Inliers (selected)", str(inlier_count))
table.add_row("Outliers (filtered)", str(outlier_count))
console.print(table)

outlier_table = Table(
    title="Detected Outliers",
    show_header=True,
    header_style="bold red",
)
outlier_table.add_column("Text", style="white")
outlier_table.add_column("Avg similarity score", style="yellow")
for record, score in zip(result.filtered, result.scores_filtered):
    outlier_table.add_row(record["text"], f"{score:.4f}")
console.print(outlier_table)

inliers_file = OUTPUT_DIR / "inliers.json"
outliers_file = OUTPUT_DIR / "outliers.json"

with open(inliers_file, "w") as f:
    json.dump(result.selected, f, indent=2)
with open(outliers_file, "w") as f:
    json.dump(result.filtered, f, indent=2)

console.print(
    f"[bold green]Done![/bold green] "
    f"Saved inliers to [link={inliers_file.resolve().as_uri()}]{inliers_file}[/link], "
    f"outliers to [link={outliers_file.resolve().as_uri()}]{outliers_file}[/link]"
)
