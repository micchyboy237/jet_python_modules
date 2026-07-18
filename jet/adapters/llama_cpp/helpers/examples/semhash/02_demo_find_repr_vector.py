"""
02_demo_find_repr_vector.py

Demonstrates SemHash.from_embeddings(): building a SemHash index from
*pre-computed* embeddings (produced by our local llama.cpp embedding server)
instead of letting SemHash compute them internally via SemHash.from_records().

Why this matters:
- SemHash.from_records() always calls model.encode() itself, computing
  embeddings from scratch every time.
- SemHash.from_embeddings() lets you bring your own embeddings (already
  computed, cached, or produced by an external service like our llama.cpp
  server), while still passing a compatible `model` so SemHash can re-encode
  small batches internally later. That re-encode happens during the
  diversity re-ranking step (self_find_representative -> _diversify), which
  only runs on the shortlisted candidates, not the full dataset.
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

    SemHash calls `model.encode(inputs)` internally during the diversity
    re-ranking step (`_diversify`), so this wrapper must work standalone on
    plain text lists, not just at initial index-build time.
    """

    def encode(self, inputs, **kwargs) -> np.ndarray:
        """
        Encode a string or list of strings into embeddings.
        :param inputs: A single string or a list of strings to encode.
        :param **kwargs: Ignored. Kept for Encoder protocol compatibility.
        :return: Embeddings as a numpy array of shape (n_inputs, embedding_dim).
        """
        texts = [inputs] if isinstance(inputs, str) else list(inputs)
        console.log(f"[LlamaCppEncoder] Encoding {len(texts)} text(s)...")
        embeddings = embed(texts, return_format="numpy", show_progress=False)
        console.log(f"[LlamaCppEncoder] Done. Output shape: {np.shape(embeddings)}")
        return embeddings


BASE_DIR = os.path.join(os.path.dirname(__file__), "mocks")
sample_data_path = os.path.join(BASE_DIR, "02_data.json")

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

with console.status("[bold green]Finding representative samples...", spinner="dots"):
    result = semhash.self_find_representative(diversity=0.5)

representative_count = len(result.selected)
table = Table(
    title="Representative Samples Results (from precomputed embeddings)",
    show_header=True,
    header_style="bold magenta",
)
table.add_column("Metric", style="cyan")
table.add_column("Value", style="green")
table.add_row("Input records", str(len(sample_records)))
table.add_row("Representative count", str(representative_count))
table.add_row(
    "Reduction ratio", f"{1 - (representative_count / len(sample_records)):.2%}"
)
console.print(table)

output_file = OUTPUT_DIR / "representative_data.json"
with open(output_file, "w") as f:
    json.dump(result.selected, f, indent=2)

output_uri = output_file.resolve().as_uri()
console.print(
    f"[bold green]Done![/bold green] Saved representative output to [link={output_uri}]{output_file}[/link]"
)
