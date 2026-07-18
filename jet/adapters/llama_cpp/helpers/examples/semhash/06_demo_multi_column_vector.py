"""
06_demo_multi_column_vector.py

Demonstrates SemHash with MULTIPLE columns AND a custom encoder
(llama.cpp server via embed_utils), instead of the default SemHash model.

Critical detail (see semhash/utils.py: featurize()):
  SemHash encodes each column SEPARATELY, then concatenates the resulting
  embeddings side-by-side:
      final_vector = [ encode(titles) | encode(texts) ]
  It does NOT concatenate the raw text first and then encode once.

So when pre-computing embeddings for from_embeddings(), we must replicate
this exact per-column encode-then-concatenate pattern ourselves — otherwise
our embeddings won't be compatible with what SemHash computes internally
during self_find_representative()'s diversity re-ranking step (_diversify
calls featurize() again on the shortlisted candidates, encoding each column
separately through our custom encoder).
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

    For multi-column records, SemHash calls this .encode() once PER COLUMN
    (both at index-build time here, and again during _diversify()'s
    re-ranking step), passing only that column's list of texts each time.
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


def compute_multi_column_embeddings(
    records: list[dict], columns: list[str], encoder: LlamaCppEncoder
) -> np.ndarray:
    """
    Pre-compute embeddings for multi-column records, mirroring SemHash's own
    featurize() logic exactly: encode each column separately, then
    concatenate the resulting embeddings side-by-side (axis=1).

    This ensures embeddings passed to SemHash.from_embeddings() are shaped
    and ordered identically to what SemHash would produce internally.

    :param records: List of dict records containing all specified columns.
    :param columns: Column names to featurize, in order.
    :param encoder: The encoder used to embed each column's texts.
    :return: Concatenated embeddings of shape (n_records, sum_of_column_dims).
    """
    embeddings_per_col = []
    for col in columns:
        col_texts = [r[col] for r in records]
        console.print(
            f"[bold]Embedding column:[/bold] '{col}' ({len(col_texts)} texts)"
        )
        col_emb = encoder.encode(col_texts)
        embeddings_per_col.append(np.asarray(col_emb))
    return np.concatenate(embeddings_per_col, axis=1)


BASE_DIR = os.path.join(os.path.dirname(__file__), "mocks")
sample_data_path = os.path.join(BASE_DIR, "06_data.json")

with console.status("[bold green]Loading sample data...", spinner="dots"):
    with open(sample_data_path, "r") as f:
        sample_records = json.load(f)
console.print(f"Loaded [bold]{len(sample_records)}[/bold] sample records")

COLUMNS = ["title", "text"]
console.log(f"Using columns: {COLUMNS}")

encoder = LlamaCppEncoder()

console.print(
    "[bold]Pre-computing per-column embeddings via llama.cpp server...[/bold]"
)
doc_embeddings = compute_multi_column_embeddings(sample_records, COLUMNS, encoder)
console.print(
    f"[green]Computed concatenated embeddings[/green] with shape [bold]{np.shape(doc_embeddings)}[/bold]"
)

with console.status(
    "[bold green]Building SemHash index from precomputed multi-column embeddings...",
    spinner="dots",
):
    semhash = SemHash.from_embeddings(
        embeddings=doc_embeddings,
        records=sample_records,
        model=encoder,
        columns=COLUMNS,
    )
console.print("[green]SemHash index built from precomputed embeddings[/green]")

with console.status("[bold green]Finding representative samples...", spinner="dots"):
    result = semhash.self_find_representative(diversity=0.5)

representative_count = len(result.selected)
table = Table(
    title="Representative Samples Results (multi-column, precomputed embeddings)",
    show_header=True,
    header_style="bold magenta",
)
table.add_column("Metric", style="cyan")
table.add_column("Value", style="green")
table.add_row("Input records", str(len(sample_records)))
table.add_row("Columns used", ", ".join(COLUMNS))
table.add_row("Representative count", str(representative_count))
table.add_row(
    "Reduction ratio", f"{1 - (representative_count / len(sample_records)):.2%}"
)
console.print(table)

repr_table = Table(title="Selected Representatives", header_style="bold cyan")
repr_table.add_column("Title", style="white")
repr_table.add_column("Text", style="white")
for record in result.selected:
    repr_table.add_row(record["title"], record["text"])
console.print(repr_table)

output_file = OUTPUT_DIR / "representative_data.json"
with open(output_file, "w") as f:
    json.dump(result.selected, f, indent=2)

output_uri = output_file.resolve().as_uri()
console.print(
    f"[bold green]Done![/bold green] Saved representative output to [link={output_uri}]{output_file}[/link]"
)
