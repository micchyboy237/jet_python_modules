"""Demo 05: Grammar-constrained output via GBNF."""

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

BOOK_REVIEW_GRAMMAR = """\
root ::= "{" ws "\\"title\\"" ws ":" ws string ws "," ws "\\"rating\\"" ws ":" ws number ws "," ws "\\"verdict\\"" ws ":" ws verdict ws "}"
verdict ::= "\\"RECOMMEND\\"" | "\\"NEUTRAL\\"" | "\\"AVOID\\""
string ::= "\\"" ([^"\\\\] | "\\\\" .){1,200} "\\""
number ::= [1-5]
ws ::= [ \\t\\n]*
"""


def demo_grammar_mode():
    console.print(
        Panel.fit(
            "📜 [bold]Demo 05: Grammar-Constrained Output (GBNF)[/bold]\n"
            "Token-level constraints guarantee valid JSON every time.",
            style="blue",
        )
    )
    setup_observability(project_name="demo-grammar")
    client = get_client()

    prompt = (
        "Write a brief review of 'Dune' by Frank Herbert.\n"
        "Respond with JSON: title (string), rating (1-5), verdict (RECOMMEND/NEUTRAL/AVOID).\n"
        "Return ONLY JSON."
    )

    t0 = time.perf_counter()
    result = run_chat_stream(
        prompt,
        client=client,
        temperature=0.0,
        max_tokens=300,
        enable_thinking=False,
        response_format={"type": "grammar", "grammar": BOOK_REVIEW_GRAMMAR},
    )
    duration_ms = (time.perf_counter() - t0) * 1000

    console.print("\n[bold green]✅ Grammar Result:[/bold green]")
    console.print(f"   [dim]{result.content}[/dim]")

    try:
        parsed = json.loads(result.content)
        console.print(f"\n   [cyan]Parsed (guaranteed valid):[/cyan]")
        console.print_json(json.dumps(parsed, indent=2))
    except json.JSONDecodeError as e:
        console.print(f"\n   [red]❌ Parse failure: {e}[/red]")
        return

    console.print(f"\n   [dim]Duration: {duration_ms:.0f}ms[/dim]")
    if result.usage:
        console.print(
            f"   [dim]Tokens: {result.usage['prompt_tokens']} + "
            f"{result.usage['completion_tokens']} = {result.usage['total_tokens']}[/dim]"
        )


def demo_comparison_table():
    table = Table(title="Structured Output Approaches")
    table.add_column("Feature", style="cyan")
    table.add_column("json_object", style="yellow")
    table.add_column("Pydantic", style="green")
    table.add_column("Grammar", style="bold magenta")
    table.add_row("Reliability", "Best-effort", "Schema-prompted", "✅ Guaranteed")
    table.add_row("Post-parse", "Yes", "Yes + validate", "No")
    table.add_row("Thinking OK", "✅", "✅", "❌")
    table.add_row("Setup", "None", "Define model", "Write GBNF")
    table.add_row("Best for", "Prototyping", "Type-safe apps", "Agents")
    console.print("\n")
    console.print(table)


def main():
    demo_grammar_mode()
    demo_comparison_table()


if __name__ == "__main__":
    main()
