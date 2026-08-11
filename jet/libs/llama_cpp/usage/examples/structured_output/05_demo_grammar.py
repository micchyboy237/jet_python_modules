"""Demo 05: Grammar-constrained output via GBNF.
Demonstrates guaranteed-valid structured output using llama.cpp's
native grammar engine. Unlike json_object mode (best-effort), grammar
mode constrains token sampling so the output ALWAYS matches the schema.

What this shows:
  - Passing GBNF grammar via extra_body_params in run_chat_stream()
  - Why enable_thinking MUST be False when using grammars
  - Guaranteed valid JSON without regex extraction or retries
  - Comparing grammar mode vs json_object mode reliability
  - How webswarm uses this pattern for planner/searcher/compressor nodes
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from jet.libs.llama_cpp.usage.chat_stream_observability import (
    get_client,
    run_chat_stream,
    setup_observability,
)
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, markup=True, rich_tracebacks=True)],
)
logger = logging.getLogger(Path(__file__).stem)

# Simple GBNF grammar for a book review.
# This guarantees the output is always valid JSON matching this exact structure.
BOOK_REVIEW_GRAMMAR = """\
root ::= "{" ws "\\"title\\"" ws ":" ws string ws "," ws "\\"rating\\"" ws ":" ws number ws "," ws "\\"verdict\\"" ws ":" ws verdict ws "}"
verdict ::= "\\"RECOMMEND\\"" | "\\"NEUTRAL\\"" | "\\"AVOID\\""
string ::= "\\"" ([^"\\\\] | "\\\\" .){1,200} "\\""
number ::= [1-5]
ws ::= [ \\t\\n]*
"""


def demo_grammar_mode():
    """Run a grammar-constrained completion."""
    console.print(
        Panel.fit(
            "📜 [bold]Demo 05: Grammar-Constrained Output (GBNF)[/bold]\n"
            "Token-level constraints guarantee valid JSON every time.\n"
            "No regex extraction, no retries, no markdown fences.",
            style="blue",
        )
    )

    setup_observability(project_name="demo-grammar")
    client = get_client()

    prompt = (
        "Write a brief review of the book 'Dune' by Frank Herbert.\n"
        "You MUST respond with a JSON object containing exactly:\n"
        '  - "title": string (book title)\n'
        '  - "rating": integer 1-5\n'
        '  - "verdict": one of "RECOMMEND", "NEUTRAL", "AVOID"\n'
        "Return ONLY the JSON object."
    )

    console.print("\n[bold yellow]⚙️  Grammar mode requires:[/bold yellow]")
    console.print("   • extra_body.grammar = GBNF string")
    console.print("   • enable_thinking = False (thinking tokens break grammar)")
    console.print("   • stream = True (accumulated then parsed)\n")

    t0 = time.perf_counter()

    # Key difference from other demos: grammar goes in extra_body_params,
    # NOT in response_format. enable_thinking MUST be False.
    result = run_chat_stream(
        client=client,
        prompt=prompt,
        temperature=0.0,
        max_tokens=300,
        enable_thinking=False,
        extra_body_params={
            "grammar": BOOK_REVIEW_GRAMMAR,
        },
    )

    duration_ms = (time.perf_counter() - t0) * 1000

    console.print("\n[bold green]✅ Grammar Result:[/bold green]")
    console.print(f"   [yellow]Raw response:[/yellow]")
    console.print(f"   [dim]{result.content}[/dim]")

    # With grammar mode, json.loads should NEVER fail.
    # No extract_json() needed — output is guaranteed valid.
    try:
        parsed = json.loads(result.content)
        console.print(f"\n   [cyan]Parsed JSON (guaranteed valid):[/cyan]")
        console.print_json(json.dumps(parsed, indent=2))
    except json.JSONDecodeError as e:
        console.print(f"\n   [red]❌ Unexpected parse failure: {e}[/red]")
        console.print("   [red]This indicates a grammar definition bug.[/red]")
        return

    console.print(f"\n   [dim]Duration: {duration_ms:.0f}ms[/dim]")
    if result.usage:
        console.print(
            f"   [dim]Tokens: {result.usage['prompt_tokens']} + "
            f"{result.usage['completion_tokens']} = "
            f"{result.usage['total_tokens']}[/dim]"
        )


def demo_comparison_table():
    """Show a comparison of structured output approaches."""
    table = Table(title="Structured Output Approaches Comparison")
    table.add_column("Feature", style="cyan", no_wrap=True)
    table.add_column("json_object", style="yellow")
    table.add_column("Pydantic", style="green")
    table.add_column("Grammar (GBNF)", style="bold magenta")

    table.add_row("Reliability", "Best-effort", "Schema-prompted", "✅ Guaranteed")
    table.add_row("Post-parse needed", "Yes (regex)", "Yes + validate", "No")
    table.add_row("Thinking mode", "Supported", "Supported", "❌ Must disable")
    table.add_row("Setup complexity", "None", "Define model", "Write GBNF")
    table.add_row("Flexibility", "Any JSON", "Any Pydantic model", "Custom schemas")
    table.add_row("Best for", "Prototyping", "Type-safe apps", "Agents / pipelines")
    table.add_row("Used in webswarm?", "No", "No", "✅ Planner/Searcher")

    console.print("\n")
    console.print(table)


def main():
    demo_grammar_mode()
    demo_comparison_table()

    console.print(
        "\n[dim]💡 Tip: See examples/webswarm/grammars/ for production GBNF files[/dim]"
    )
    console.print(
        "[dim]   used by planner, searcher, compressor, and confidence nodes.[/dim]"
    )


if __name__ == "__main__":
    main()
