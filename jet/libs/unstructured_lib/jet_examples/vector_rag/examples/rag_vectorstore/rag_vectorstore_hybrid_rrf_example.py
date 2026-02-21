"""Example: hybrid vector + BM25 with Reciprocal Rank Fusion demo."""

import json
import shutil
from datetime import datetime
from pathlib import Path

from jet.libs.unstructured_lib.jet_examples.vector_rag.examples.rag_vectorstore.common import (
    DOCUMENTS,
)
from jet.libs.unstructured_lib.jet_examples.vector_rag.rag_pipeline import RAGPipeline
from rich.console import Console

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


console.rule("Hybrid Vector + BM25 (RRF) Retrieval Example")

pipeline = RAGPipeline()

temp_files = []

for name, content in DOCUMENTS.items():
    temp_path = OUTPUT_DIR / f"ingest_{name}"
    temp_path.write_text(content)
    temp_files.append(temp_path)
    console.print(f"[blue]Ingesting[/blue] {name}")
    pipeline.ingest(str(temp_path))

save_json(
    {
        "ingested_files": len(DOCUMENTS),
        "timestamp": datetime.now().isoformat(),
        "retrieval_mode": "hybrid-rrf",
        "pipeline_config": {"k_default": 4, "oversample_factor": 3},
    },
    "run_summary",
)

for name, content in DOCUMENTS.items():
    save_text(content, f"input_document_{name}")

console.rule("Running Mixed Queries")

queries = [
    {
        "text": "calculate_risk_score formula",
        "note": "should benefit from keyword match",
    },
    {
        "text": "What are the main benefits of asyncio over threading?",
        "note": "semantic – vector strong",
    },
    {
        "text": "Grok-4 context length and release date",
        "note": "hybrid should combine both",
    },
    {
        "text": "refund policy for software licenses",
        "note": "exact phrase → BM25 helps",
    },
    {
        "text": "How to handle timeouts in async HTTP requests",
        "note": "conceptual + code pattern",
    },
]

for idx, q in enumerate(queries, 1):
    console.print(
        f"\n[bold green]Query {idx}:[/] {q['text']}  [dim]({q.get('note', '')})[/]"
    )
    answer = pipeline.query(q["text"], k=4, temperature=0.0, mode="hybrid-rrf")

    save_text(q["text"], f"query_{idx:02d}_question")
    save_text(answer, f"query_{idx:02d}_answer_hybrid_rrf")

    console.print(f"[cyan]Answer:[/]\n{answer}\n")

for p in temp_files:
    if p.exists():
        p.unlink()

console.print(f"\n[green]Artifacts saved to:[/] {OUTPUT_DIR}")
