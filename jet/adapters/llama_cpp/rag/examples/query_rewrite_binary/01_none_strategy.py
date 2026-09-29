"""
Demo: NONE Strategy (Skip Rewriting)
Covers: Simple factual queries that bypass LLM rewriting for maximum speed.
Span Hierarchy:
    📦 none-strategy-demo (CHAIN)
    │
    └── 🧠 analyze_query_intent (LLM)
        ├── attr: llm.model_name = "qwen3.5-uncensored:2b"
        └── attr: llm.input_messages = [...]
"""

import shutil
from pathlib import Path

from jet.adapters.llama_cpp.config import PHOENIX_BASE_URL
from jet.adapters.llama_cpp.rag.query_rewrite_binary import rewrite_query
from jet_telemetry import (
    chain,
    export_spans_to_jsonl,
    get_trace_url,
    initialize_telemetry,
)
from rich.console import Console
from rich.panel import Panel

console = Console()
OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PROJECT_NAME = "binary-query-rewrite"


@chain(name="none-strategy-demo")
def run_demo():
    """Demonstrate skipping rewriting for simple queries."""
    query = "What is Python?"

    console.print(
        Panel(
            f"[bold]Query:[/bold] {query}\n[dim]Expected: NONE (Simple/Factual)[/dim]",
            title="🚀 Demo 1: None Strategy",
            border_style="green",
        )
    )

    result = rewrite_query(query=query, output_dir=OUTPUT_DIR)

    console.print(f"\n[bold cyan]Strategy:[/bold cyan] {result.strategy}")
    console.print(f"[bold cyan]Reason:[/bold cyan] {result.reason}")
    console.print(f"[bold cyan]Queries:[/bold cyan] {result.queries}")

    assert result.strategy == "NONE", "Expected NONE strategy for simple query"
    assert result.queries == [query], "Original query should be preserved"

    # Export traces explicitly
    trace_url = get_trace_url(PHOENIX_BASE_URL)
    if trace_url:
        console.print(f"\n[dim]Trace URL: {trace_url}[/dim]")
        try:
            # Extract trace ID from URL for export
            trace_id = trace_url.split("/redirects/traces/")[-1]
            jsonl_path = export_spans_to_jsonl(
                project_name=PROJECT_NAME,  # Use the project name from the internal calls
                trace_id=trace_id,
                output_path=OUTPUT_DIR / f"{trace_id}.jsonl",
                phoenix_base_url=PHOENIX_BASE_URL,
                wait_for_flush=True,
            )
            console.print(f"[green]✅ Traces exported to: {jsonl_path}[/green]")
        except Exception as e:
            console.print(f"[yellow]⚠️ Could not auto-export JSONL: {e}[/yellow]")


if __name__ == "__main__":
    initialize_telemetry(service_name=PROJECT_NAME, endpoint=PHOENIX_BASE_URL)
    run_demo()
