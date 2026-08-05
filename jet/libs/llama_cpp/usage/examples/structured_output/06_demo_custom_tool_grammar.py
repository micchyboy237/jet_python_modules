"""Demo 06: Custom tool with grammar format (OpenAI official pattern).

Demonstrates the official OpenAI CustomFormatGrammar tool type.
This is the standard way to provide grammar constraints per the OpenAI API spec.
Uses ChatCompletionCustomToolParam → Custom → CustomFormatGrammar.

What this shows:
  - custom_tool_grammar_output() with regex grammar
  - Building a custom tool definition with grammar format
  - The difference between this and function calling
  - How to access: result.tool_calls, result.parsed
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from jet.libs.llama_cpp.usage.chat_stream_observability import (
    get_client,
    setup_observability,
)
from jet.libs.llama_cpp.usage.structured_output import (
    StructuredResult,
    custom_tool_grammar_output,
)
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, markup=True, rich_tracebacks=True)],
)
logger = logging.getLogger(Path(__file__).stem)


def main():
    console.print(
        Panel.fit(
            "🔧 [bold]Demo 06: Custom Tool Grammar (OpenAI Official)[/bold]\n"
            "Uses the official OpenAI CustomFormatGrammar tool pattern.\n"
            "Grammar syntax: regex or lark (per OpenAI spec).",
            style="blue",
        )
    )

    setup_observability(project_name="demo-custom-tool-grammar")
    client = get_client()

    # ─── Single complete example ──────────────────────────────────────
    # Define a regex grammar for a simple contact info structure
    # This constrains output to: "Name: <text>, Email: <email>, Phone: <digits>"
    contact_grammar = (
        r"Name:\s*[A-Z][a-z]+(?:\s[A-Z][a-z]+)*,\s*"
        r"Email:\s*[\w.+-]+@[\w-]+\.[\w.]+,\s*"
        r"Phone:\s*\+?[\d\s()-]{7,15}"
    )

    prompt = (
        "Generate contact information for a fictional person named "
        "Sarah Johnson with email sarah.j@example.com and phone "
        "+1-555-0123. Use the format: Name, Email, Phone."
    )

    result: StructuredResult = custom_tool_grammar_output(
        client,
        prompt=prompt,
        grammar_definition=contact_grammar,
        tool_name="format_contact",
        tool_description="Format contact information in a specific structure",
        grammar_syntax="regex",
        temperature=0.0,
        max_tokens=200,
    )

    # ─── Inspect the result ───────────────────────────────────────────
    console.print("\n[bold green]✅ Result:[/bold green]")

    if result.success and result.tool_calls:
        for tc in result.tool_calls:
            console.print(f"   [cyan]Tool:[/cyan] {tc['name']}")
            console.print(
                f"   [cyan]Output:[/cyan] {json.dumps(tc['arguments'], indent=2)}"
            )

        if result.parsed:
            console.print(
                f"   [cyan]Parsed:[/cyan] {json.dumps(result.parsed, indent=2)}"
            )
    elif result.content:
        console.print(f"   [yellow]Raw output (no tool call):[/yellow]")
        console.print(f"   [dim]{result.content[:300]}[/dim]")
    else:
        console.print(f"   [red]Failed: {result.error}[/red]")

    console.print(f"\n   [dim]Format used: {result.format_used.value}[/dim]")
    console.print(f"   [dim]Duration: {result.duration_ms:.0f}ms[/dim]")

    if result.usage:
        console.print(
            f"   [dim]Tokens: {result.usage['prompt_tokens']} + "
            f"{result.usage['completion_tokens']} = "
            f"{result.usage['total_tokens']}[/dim]"
        )


if __name__ == "__main__":
    main()
