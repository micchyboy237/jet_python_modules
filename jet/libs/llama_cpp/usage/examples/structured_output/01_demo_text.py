"""Demo 01: Plain text chat completion with Phoenix observability.

Demonstrates the simplest form of LLM interaction — text in, text out.
Always works on any llama.cpp model.

What this shows:
  - Basic text streaming via text_output()
  - StructuredResult return type with usage stats
  - Phoenix observability integration
  - How to access: result.content, result.usage, result.finish_reason
"""

from __future__ import annotations

import logging
from pathlib import Path

from jet.libs.llama_cpp.usage.chat_stream_observability import (
    get_client,
    setup_observability,
)
from jet.libs.llama_cpp.usage.structured_output import (
    StructuredResult,
    text_output,
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
            "📝 [bold]Demo 01: Plain Text Output[/bold]\n"
            "The simplest form — text in, text out. Always works.",
            style="blue",
        )
    )

    setup_observability(project_name="demo-text")
    client = get_client()

    # ─── Single complete example ──────────────────────────────────────
    prompt = (
        "Explain what structured output means for LLMs in 2-3 sentences. "
        "Be concise and clear."
    )

    result: StructuredResult = text_output(
        client,
        prompt=prompt,
        temperature=0.3,
        max_tokens=200,
    )

    # ─── Inspect the result ───────────────────────────────────────────
    console.print("\n[bold green]✅ Result:[/bold green]")
    console.print(f"   [cyan]Content:[/cyan] {result.content}")
    console.print(f"   [dim]Format used: {result.format_used.value}[/dim]")
    console.print(f"   [dim]Duration: {result.duration_ms:.0f}ms[/dim]")
    console.print(f"   [dim]Finish reason: {result.finish_reason}[/dim]")

    if result.usage:
        console.print(
            f"   [dim]Tokens: {result.usage['prompt_tokens']} prompt + "
            f"{result.usage['completion_tokens']} completion = "
            f"{result.usage['total_tokens']} total[/dim]"
        )


if __name__ == "__main__":
    main()
