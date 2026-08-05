"""Demo 04: OpenAI-compatible parsed completion pattern.

Demonstrates parsed_completion() which mimics the OpenAI SDK's
pydantic_function_tool() pattern. The interface is the same:
  result.parsed → your Pydantic model instance

This is the closest drop-in replacement for:
  client.chat.completions.create(
      ...,
      response_format=pydantic_function_tool(MyModel),
  )
  model = result.choices[0].message.parsed

What this shows:
  - parsed_completion() with .parsed attribute
  - Same interface as OpenAI's pydantic_function_tool()
  - Direct model access: result.parsed.field_name
  - Compares to the official OpenAI pattern
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
    ParsedOutput,
    parsed_completion,
)
from pydantic import BaseModel, Field
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.syntax import Syntax

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, markup=True, rich_tracebacks=True)],
)
logger = logging.getLogger(Path(__file__).stem)


# ─── Define the Pydantic model once ────────────────────────────────────────


class Conference(BaseModel):
    """Information about a tech conference."""

    name: str = Field(description="Conference name")
    location: str = Field(description="City and country")
    year: int = Field(description="Year of the conference")
    attendees: int = Field(description="Approximate number of attendees")
    topics: list[str] = Field(description="Main topics covered")


def main():
    console.print(
        Panel.fit(
            "🔄 [bold]Demo 04: OpenAI-Compatible parsed_completion()[/bold]\n"
            "Mimics OpenAI's pydantic_function_tool() interface.\n"
            "Access with: result.parsed.field_name",
            style="blue",
        )
    )

    # Show the OpenAI pattern this replicates
    console.print("[dim]This replicates the OpenAI SDK pattern:[/dim]")
    console.print(
        Syntax(
            "# OpenAI SDK (works only with OpenAI API):\n"
            "# result = client.beta.chat.completions.parse(\n"
            "#     model='gpt-4o',\n"
            "#     messages=[{'role': 'user', 'content': '...'}],\n"
            "#     response_format=Conference,\n"
            "# )\n"
            "# conference = result.choices[0].message.parsed\n"
            "\n"
            "# Our llama.cpp equivalent (same interface):\n"
            "result = parsed_completion(client, prompt, Conference)\n"
            "conference = result.parsed  # ← Same .parsed access!",
            "python",
            theme="monokai",
            line_numbers=False,
        )
    )

    setup_observability(project_name="demo-parsed-completion")
    client = get_client()

    # ─── Single complete example ──────────────────────────────────────
    text = (
        "PyCon US 2024 was held in Pittsburgh, Pennsylvania. "
        "It attracted over 2,500 attendees and covered topics including "
        "machine learning, web development, DevOps, and core Python language "
        "features."
    )

    result: ParsedOutput[Conference] = parsed_completion(
        client,
        f"Extract conference information as JSON:\n{text}",
        Conference,
        temperature=0.0,
        max_tokens=300,
    )

    # ─── Inspect the result ───────────────────────────────────────────
    console.print("\n[bold green]✅ Result:[/bold green]")

    if result.parsed:
        # Same interface as OpenAI's .parsed
        conference = result.parsed
        console.print(f"   [cyan]Name:[/cyan] {conference.name}")
        console.print(f"   [cyan]Location:[/cyan] {conference.location}")
        console.print(f"   [cyan]Year:[/cyan] {conference.year}")
        console.print(f"   [cyan]Attendees:[/cyan] {conference.attendees:,}")
        console.print(f"   [cyan]Topics:[/cyan] {', '.join(conference.topics)}")

        console.print(f"\n   [dim]Full parsed model:[/dim]")
        console.print_json(json.dumps(conference.model_dump(), indent=2, default=str))
    else:
        console.print(f"   [red]Parsing failed - no model extracted[/red]")

    console.print(f"\n   [dim]Content length: {len(result.content)} chars[/dim]")
    console.print(f"   [dim]Finish reason: {result.finish_reason}[/dim]")

    if result.usage:
        console.print(
            f"   [dim]Tokens: {result.usage['prompt_tokens']} + "
            f"{result.usage['completion_tokens']} = "
            f"{result.usage['total_tokens']}[/dim]"
        )


if __name__ == "__main__":
    main()
