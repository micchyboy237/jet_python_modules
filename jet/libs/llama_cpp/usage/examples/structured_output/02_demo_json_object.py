"""Demo 02: JSON object output via response_format.

Demonstrates best-effort JSON mode on llama.cpp.
The model is hinted to return JSON but may wrap it in ``` fences.
Our extract_json() helper handles this automatically.

What this shows:
  - json_object_output() with response_format={"type": "json_object"}
  - Automatic JSON extraction from markdown fences
  - How to access: result.parsed (dict/list), result.content (raw str)
  - Comparing raw content vs parsed JSON
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
    json_object_output,
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
            "🏗️ [bold]Demo 02: JSON Object Output[/bold]\n"
            "Best-effort JSON via response_format={'type': 'json_object'}.\n"
            "Handles markdown fences automatically.",
            style="blue",
        )
    )

    setup_observability(project_name="demo-json-object")
    client = get_client()

    # ─── Single complete example ──────────────────────────────────────
    prompt = (
        "Extract the key facts about Python from this text as a JSON object:\n\n"
        "Python was created by Guido van Rossum in 1991. "
        "It is an interpreted, high-level programming language known for "
        "its readability and versatility. The latest major version is Python 3.13.\n\n"
        "Return a JSON object with these fields:\n"
        '  - "creator" (string)\n'
        '  - "year_created" (number)\n'
        '  - "type" (string)\n'
        '  - "known_for" (array of strings)\n'
        '  - "latest_version" (string)\n'
        "Return ONLY the JSON object, no markdown, no explanation."
    )

    result: StructuredResult = json_object_output(
        client,
        prompt=prompt,
        temperature=0.0,
        max_tokens=300,
    )

    # ─── Inspect the result ───────────────────────────────────────────
    console.print("\n[bold green]✅ Result:[/bold green]")

    # Show raw content (may have markdown fences)
    console.print(f"   [yellow]Raw response:[/yellow]")
    console.print(f"   [dim]{result.content[:200]}[/dim]")

    # Show parsed JSON (extracted & cleaned)
    if result.success and result.parsed:
        console.print(f"\n   [cyan]Parsed JSON:[/cyan]")
        console.print_json(json.dumps(result.parsed, indent=2))
    else:
        console.print(f"\n   [red]Failed to parse: {result.error}[/red]")

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
