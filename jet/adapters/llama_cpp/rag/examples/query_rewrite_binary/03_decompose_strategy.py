"""
Demo: DECOMPOSE Strategy (Query Breakdown)
Covers: Complex multi-part queries broken into independent sub-queries for parallel retrieval.
Span Hierarchy:
    📦 decompose-strategy-demo (CHAIN)
    │
    ├── 🧠 analyze_query_intent (LLM)
    │   ├── attr: llm.model_name = "qwen3.5-uncensored:2b"
    │   └── attr: llm.input_messages = [...]
    │
    └── 🛠️ decompose_query (TOOL)
        ├── attr: tool.name = "decompose_query"
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


@chain(name="decompose-strategy-demo")
def run_demo():
    """Demonstrate decomposing a complex comparative query."""
    query = "Compare the security features, pricing models, and scalability of AWS vs Azure for enterprise ML workloads."

    console.print(
        Panel(
            f"[bold]Query:[/bold] {query[:80]}...\n[dim]Expected: DECOMPOSE (Complex/Multi-part)[/dim]",
            title="⚡ Demo 3: Decompose Strategy",
            border_style="red",
        )
    )

    result = rewrite_query(query=query, output_dir=OUTPUT_DIR)

    table = Table(
        title="Decomposition Results", show_header=True, header_style="bold magenta"
    )
    table.add_column("#", style="cyan", width=3)
    table.add_column("Sub-Query", style="green")

    for i, q in enumerate(result.queries, 1):
        table.add_row(str(i), q)

    console.print(f"\n[bold cyan]Strategy:[/bold cyan] {result.strategy}")
    console.print(f"[bold cyan]Reason:[/bold cyan] {result.reason}")
    console.print()
    console.print(table)

    assert result.strategy == "DECOMPOSE", (
        "Expected DECOMPOSE strategy for complex query"
    )
    assert len(result.queries) >= 2, "Decompose should return multiple sub-queries"

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
