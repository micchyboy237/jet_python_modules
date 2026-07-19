"""
07_demo_self_deduplicate_vector.py

Demonstrates SemHash.self_deduplicate(): removing duplicate records from a
single dataset, using a custom encoder (llama.cpp server via embed_utils).

Two kinds of duplicates are detected:
  - EXACT duplicates: records with identical values in the given columns.
    These are grouped together automatically at index-build time (see
    SemHash.from_embeddings() -> group_records_by_key()), before
    self_deduplicate() even runs its similarity check.
  - SEMANTIC (near) duplicates: records that are NOT identical strings, but
    whose embeddings exceed the similarity `threshold`. These are detected
    by self_deduplicate()'s query_threshold() call against the fitted index.

Each removed record is wrapped in a DuplicateRecord with an `exact` flag,
so we can show which detection path caught it.
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
sample_data_path = os.path.join(BASE_DIR, "07_data.json")

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

THRESHOLD = 0.9

with console.status("[bold green]Deduplicating dataset...", spinner="dots"):
    result = semhash.self_deduplicate(threshold=THRESHOLD)

kept_count = len(result.selected)
removed_count = len(result.filtered)
exact_count = sum(1 for dup in result.filtered if dup.exact)
semantic_count = removed_count - exact_count

summary_table = Table(
    title="Self-Deduplication Results",
    show_header=True,
    header_style="bold magenta",
)
summary_table.add_column("Metric", style="cyan")
summary_table.add_column("Value", style="green")
summary_table.add_row("Input records", str(len(sample_records)))
summary_table.add_row("Threshold", str(THRESHOLD))
summary_table.add_row("Kept (unique)", str(kept_count))
summary_table.add_row("Removed (exact duplicates)", str(exact_count))
summary_table.add_row("Removed (semantic duplicates)", str(semantic_count))
console.print(summary_table)

removed_table = Table(
    title="Removed Duplicate Records",
    show_header=True,
    header_style="bold red",
)
removed_table.add_column("Removed Text", style="white")
removed_table.add_column("Type", style="yellow")
removed_table.add_column("Matched Against", style="white")
for dup_record in result.filtered:
    matched_texts = (
        ", ".join(d["text"] for d, _score in dup_record.duplicates) or "(none)"
    )
    dup_type = "EXACT" if dup_record.exact else "SEMANTIC"
    removed_table.add_row(dup_record.record["text"], dup_type, matched_texts)
console.print(removed_table)

kept_table = Table(title="Kept (Unique) Records", header_style="bold cyan")
kept_table.add_column("Text", style="white")
for record in result.selected:
    kept_table.add_row(record["text"])
console.print(kept_table)

selected_file = OUTPUT_DIR / "deduplicated_data.json"
with open(selected_file, "w") as f:
    json.dump(result.selected, f, indent=2)

output_uri = selected_file.resolve().as_uri()
console.print(
    f"[bold green]Done![/bold green] Saved deduplicated output to [link={output_uri}]{selected_file}[/link]"
)
