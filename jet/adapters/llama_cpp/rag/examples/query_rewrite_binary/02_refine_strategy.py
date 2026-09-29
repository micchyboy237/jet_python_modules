"""
Demo: REFINE Strategy (Query Clarification)
Covers: Ambiguous or context-dependent queries rewritten into a single specific query.
Span Hierarchy:
    📦 refine-strategy-demo (CHAIN)
    │
    ├── 🧠 analyze_query_intent (LLM)
    │   ├── attr: llm.model_name = "qwen3.5-uncensored:2b"
    │   └── attr: llm.input_messages = [...]
    │
    └── 🛠️ refine_query (TOOL)
        ├── attr: tool.name = "refine_query"
        └── attr: tool.parameters = {"original_query": "..."}
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
from rich.table import Table

console = Console()
OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PROJECT_NAME = "binary-query-rewrite"


@chain(name="refine-strategy-demo")
def run_demo():
    """Demonstrate refining a vague query with context."""
    query = "How does it work?"
    context = "Previous topic: Kubernetes Pod Disruption Budgets"

    console.print(
        Panel(
            f"[bold]Query:[/bold] {query}\n[bold]Context:[/bold] {context}\n[dim]Expected: REFINE (Ambiguous/Contextual)[/dim]",
            title="🎯 Demo 2: Refine Strategy",
            border_style="yellow",
        )
    )

    result = rewrite_query(query=query, context=context, output_dir=OUTPUT_DIR)

    table = Table(
        title="Rewrite Results", show_header=True, header_style="bold magenta"
    )
    table.add_column("Attribute", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Strategy", result.strategy)
    table.add_row("Reason", result.reason)
    table.add_row("Original", result.original_query)
    table.add_row("Refined Query", result.queries[0])

    console.print()
    console.print(table)

    assert result.strategy == "REFINE", "Expected REFINE strategy for ambiguous query"
    assert len(result.queries) == 1, "Refine should return exactly one query"

    # Export traces explicitly
    trace_url = get_trace_url(PHOENIX_BASE_URL)
    if trace_url:
        console.print(f"\n[dim]Trace URL: {trace_url}[/dim]")
        try:
            trace_id = trace_url.split("/redirects/traces/")[-1]
            jsonl_path = export_spans_to_jsonl(
                project_name=PROJECT_NAME,
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
