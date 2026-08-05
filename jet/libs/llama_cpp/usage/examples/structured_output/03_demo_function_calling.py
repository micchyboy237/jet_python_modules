"""Demo 03: Function calling for structured extraction.

Demonstrates tool-based structured output on llama.cpp.
The model calls a defined function with typed arguments.
This is the most reliable method for structured extraction.

What this shows:
  - function_call_output() with OpenAI tool definitions
  - Tool definition with JSON Schema parameters
  - How to access: result.tool_calls (name + arguments), result.parsed
  - Forcing tool use with tool_choice="required"
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
    function_call_output,
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
            "🔧 [bold]Demo 03: Function Calling Output[/bold]\n"
            "Tool-based structured extraction via function definitions.\n"
            "Most reliable method for getting typed data from llama.cpp.",
            style="blue",
        )
    )

    setup_observability(project_name="demo-function-calling")
    client = get_client()

    # ─── Single complete example ──────────────────────────────────────
    # Define the tool with typed parameters
    tools = [
        {
            "type": "function",
            "function": {
                "name": "extract_book_info",
                "description": "Extract book information from text",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "The book title",
                        },
                        "author": {
                            "type": "string",
                            "description": "The author's full name",
                        },
                        "year": {
                            "type": "integer",
                            "description": "Year of publication",
                        },
                        "genres": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of genres",
                        },
                        "is_fiction": {
                            "type": "boolean",
                            "description": "Whether the book is fiction",
                        },
                    },
                    "required": ["title", "author", "year", "genres", "is_fiction"],
                },
            },
        }
    ]

    prompt = (
        "The book 'Dune' was written by Frank Herbert and published in 1965. "
        "It is a science fiction novel that also includes elements of adventure "
        "and political drama. It is considered one of the best-selling "
        "science fiction novels of all time."
    )

    result: StructuredResult = function_call_output(
        client,
        prompt=prompt,
        tools=tools,
        tool_choice="required",  # Force the model to call the tool
        temperature=0.0,
        max_tokens=300,
    )

    # ─── Inspect the result ───────────────────────────────────────────
    console.print("\n[bold green]✅ Result:[/bold green]")

    if result.success and result.tool_calls:
        for tc in result.tool_calls:
            console.print(f"   [cyan]Tool called:[/cyan] {tc['name']}")
            console.print(f"   [cyan]Arguments:[/cyan]")
            console.print_json(json.dumps(tc["arguments"], indent=2))
    else:
        console.print(f"   [red]No tool calls: {result.error}[/red]")
        if result.content:
            console.print(f"   [dim]Raw content: {result.content[:200]}[/dim]")

    console.print(f"\n   [dim]Format used: {result.format_used.value}[/dim]")
    console.print(f"   [dim]Duration: {result.duration_ms:.0f}ms[/dim]")
    console.print(f"   [dim]Finish reason: {result.finish_reason}[/dim]")

    if result.usage:
        console.print(
            f"   [dim]Tokens: {result.usage['prompt_tokens']} + "
            f"{result.usage['completion_tokens']} = "
            f"{result.usage['total_tokens']}[/dim]"
        )


if __name__ == "__main__":
    main()
