"""Example: pure vector (dense similarity) retrieval demo with realistic documents."""

import json
import shutil
from datetime import datetime
from pathlib import Path

from jet.libs.unstructured_lib.jet_examples.vector_rag.examples.rag_vectorstore.common import (
    DOCUMENTS,
)
from jet.libs.unstructured_lib.jet_examples.vector_rag.rag_pipeline import RAGPipeline
from rich.console import Console

# ------------------------------------------------------------------------------
# Output setup
# ------------------------------------------------------------------------------

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

console = Console()


def save_json(data: dict, filename: str) -> None:
    path = OUTPUT_DIR / f"{filename}.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    console.print(f"[dim]Saved:[/] {path.name}")


def save_text(content: str, filename: str) -> None:
    path = OUTPUT_DIR / f"{filename}.txt"
    path.write_text(content)
    console.print(f"[dim]Saved:[/] {path.name}")


# ------------------------------------------------------------------------------
# Synthetic documents (long / varied content)
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Run example
# ------------------------------------------------------------------------------

console.rule("Pure Vector Retrieval Example")

pipeline = RAGPipeline()

# Temporary files for ingestion
temp_files = []

for name, content in DOCUMENTS.items():
    temp_path = OUTPUT_DIR / f"ingest_{name}"
    temp_path.write_text(content)
    temp_files.append(temp_path)
    console.print(f"[blue]Ingesting[/blue] {name}")
    pipeline.ingest(str(temp_path))

# Save summary
save_json(
    {
        "ingested_files": len(DOCUMENTS),
        "timestamp": datetime.now().isoformat(),
        "retrieval_mode": "vector",
        "pipeline_config": {"k_default": 5},
    },
    "run_summary",
)

# Save raw documents
for name, content in DOCUMENTS.items():
    save_text(content, f"input_document_{name}")

console.rule("Running Queries")

queries = [
    {
        "text": "What is the risk calculation formula in the math utilities?",
        "expected_strength": "exact keyword + code",
    },
    {
        "text": "Explain how to use asyncio.gather with multiple tasks",
        "expected_strength": "semantic / conceptual",
    },
    {
        "text": "When was Grok-4 released and what is its context length?",
        "expected_strength": "keyword + date",
    },
    {
        "text": "What is the standard warranty period for hardware?",
        "expected_strength": "exact policy phrase",
    },
]

for idx, q in enumerate(queries, 1):
    console.print(f"\n[bold green]Query {idx}:[/] {q['text']}")
    answer = pipeline.query(q["text"], k=4, temperature=0.0, mode="vector")

    # Save per-query artifacts
    save_text(q["text"], f"query_{idx:02d}_question")
    save_text(answer, f"query_{idx:02d}_answer_vector")

    console.print(f"[cyan]Answer:[/]\n{answer}\n")

# Cleanup temp files
for p in temp_files:
    if p.exists():
        p.unlink()

console.print(f"\n[green]All artifacts saved to:[/] {OUTPUT_DIR}")
