# jet_python_modules/jet/libs/llama_cpp/usage/examples/chat_stream/07_demo_structured_output_module.py
"""Demo: Using the structured_output module for reliable JSON extraction.

Shows the new encapsulated module with:
  1. text_output() - Simple text
  2. json_object_output() - Best-effort JSON
  3. grammar_output() - Strict grammar-constrained JSON
  4. function_call_output() - Function calling for structure
  5. auto_structured() - Smart auto-selection
  6. extract_person() / extract_list() - Convenience functions

Compares success rates and shows which method to use when.
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
    GRAMMAR_TEMPLATES,
    StructuredResult,
    auto_structured,
    extract_list,
    extract_person,
    function_call_output,
    grammar_output,
    json_object_output,
    text_output,
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


def print_result(label: str, result: StructuredResult):
    """Pretty-print a StructuredResult."""
    status = "✅" if result.success else "❌"
    color = "green" if result.success else "red"

    console.print(f"\n[bold]{status} {label}[/bold]")
    console.print(f"   Format: [cyan]{result.format_used.value}[/cyan]")
    console.print(f"   Duration: [dim]{result.duration_ms:.0f}ms[/dim]")

    if result.parsed:
        console.print(f"   Parsed: [green]{json.dumps(result.parsed)}[/green]")
    elif result.tool_calls:
        for tc in result.tool_calls:
            console.print(
                f"   Tool: [green]{tc['name']}({json.dumps(tc['arguments'])})[/green]"
            )
    else:
        console.print(f"   Content: [dim]{result.content[:100]}...[/dim]")

    if result.error:
        console.print(f"   Error: [red]{result.error}[/red]")


def main():
    console.print(
        Panel.fit(
            "🧪 [bold]Structured Output Module Demo[/bold]\n"
            "Testing the new encapsulated structured_output module",
            style="blue",
        )
    )

    setup_observability(project_name="structured-output-module-demo")
    client = get_client()

    text = (
        "John Smith is a 42-year-old software engineer from San Francisco. "
        "He enjoys hiking, photography, and playing guitar."
    )

    results: list[tuple[str, StructuredResult]] = []

    # ─── Test 1: Plain text ─────────────────────────────────────────────
    console.print("\n[bold yellow]═══ Test 1: text_output() ═══[/bold yellow]")
    r = text_output(client, f"Summarize in one sentence: {text}")
    print_result("Text Output", r)
    results.append(("text_output", r))

    # ─── Test 2: JSON Object ────────────────────────────────────────────
    console.print("\n[bold yellow]═══ Test 2: json_object_output() ═══[/bold yellow]")
    r = json_object_output(
        client,
        f"Extract person info as JSON:\n{text}\n\n"
        'Return: {{"name": "...", "age": ..., "city": "..."}}',
    )
    print_result("JSON Object", r)
    results.append(("json_object", r))

    # ─── Test 3: Grammar ────────────────────────────────────────────────
    console.print("\n[bold yellow]═══ Test 3: grammar_output() ═══[/bold yellow]")
    r = grammar_output(
        client,
        f"Extract: {text}",
        grammar=GRAMMAR_TEMPLATES["person"].grammar,
        grammar_name="person",
    )
    print_result("Grammar (GBNF)", r)
    results.append(("grammar", r))

    # ─── Test 4: Function Calling ───────────────────────────────────────
    console.print("\n[bold yellow]═══ Test 4: function_call_output() ═══[/bold yellow]")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "extract_person",
                "description": "Extract person information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                        "city": {"type": "string"},
                    },
                    "required": ["name", "age", "city"],
                },
            },
        }
    ]
    r = function_call_output(client, text, tools, tool_choice="required")
    print_result("Function Calling", r)
    results.append(("function_call", r))

    # ─── Test 5: Auto-structured ────────────────────────────────────────
    console.print("\n[bold yellow]═══ Test 5: auto_structured() ═══[/bold yellow]")
    r = auto_structured(
        client,
        text,
        json_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "city": {"type": "string"},
            },
            "required": ["name", "age", "city"],
        },
    )
    print_result("Auto-Structured", r)
    results.append(("auto_structured", r))

    # ─── Test 6: Convenience functions ──────────────────────────────────
    console.print("\n[bold yellow]═══ Test 6: Convenience Functions ═══[/bold yellow]")

    r = extract_person(client, text)
    print_result("extract_person()", r)
    results.append(("extract_person", r))

    hobbies_text = "His hobbies are: hiking, photography, guitar, cooking, reading"
    r = extract_list(client, hobbies_text)
    print_result("extract_list()", r)
    results.append(("extract_list", r))

    # ─── Summary Table ──────────────────────────────────────────────────
    console.print("\n")
    table = Table(title="📊 Structured Output Method Comparison")
    table.add_column("Method", style="cyan")
    table.add_column("Format Used", style="dim")
    table.add_column("Success", style="bold")
    table.add_column("Time (ms)", justify="right")
    table.add_column("Has Data", justify="center")

    for name, result in results:
        status = "✅" if result.success else "❌"
        has_data = "✅" if (result.parsed or result.tool_calls) else "❌"
        table.add_row(
            name,
            result.format_used.value,
            status,
            f"{result.duration_ms:.0f}",
            has_data,
        )

    success_count = sum(1 for _, r in results if r.success)
    console.print(table)
    console.print(
        Panel(
            f"Overall success: [bold]{success_count}/{len(results)}[/bold] "
            f"({success_count / len(results) * 100:.0f}%)\n\n"
            "[bold]Recommendation:[/bold]\n"
            "• For reliable JSON: Use [green]grammar_output()[/green]\n"
            "• For tool-based extraction: Use [green]function_call_output()[/green]\n"
            "• For quick prototyping: Use [green]auto_structured()[/green]\n"
            "• Never use: [red]json_schema[/red] in response_format (ignored by llama.cpp)",
            style="blue",
        )
    )


if __name__ == "__main__":
    main()
