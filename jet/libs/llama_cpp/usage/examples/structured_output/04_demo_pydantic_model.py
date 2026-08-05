"""Demo 04: Pydantic model extraction with automatic validation.

Demonstrates extracting a single Pydantic model instance from text.
Uses json_object mode with schema-enhanced prompting.
The result is automatically validated against the Pydantic model.

What this shows:
  - pydantic_output() with a Pydantic BaseModel
  - Automatic JSON Schema generation from model
  - Pydantic validation of extracted data
  - How to access: result.model (Pydantic instance), result.raw_result
  - Type-safe access: result.model.name, result.model.age, etc.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from jet.libs.llama_cpp.usage.chat_stream_observability import (
    get_client,
    setup_observability,
)
from jet.libs.llama_cpp.usage.structured_output import (
    PydanticResult,
    pydantic_output,
)
from pydantic import BaseModel, Field
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


# ─── Define the Pydantic model once ────────────────────────────────────────


class ProgrammingLanguage(BaseModel):
    """Structured information about a programming language."""

    name: str = Field(description="Name of the language")
    year_created: int = Field(description="Year the language was first released")
    creator: str = Field(description="Name of the creator or organization")
    paradigm: str = Field(
        description="Programming paradigm (e.g., object-oriented, functional)"
    )
    typing: str = Field(description="Type system (e.g., static, dynamic, strong, weak)")
    popular_frameworks: list[str] = Field(
        description="List of popular frameworks or libraries"
    )
    website: Optional[str] = Field(
        default=None, description="Official website URL if mentioned"
    )


def main():
    console.print(
        Panel.fit(
            "🔒 [bold]Demo 04: Pydantic Model Extraction[/bold]\n"
            "Single Pydantic model with automatic validation.\n"
            "Type-safe access: result.model.name, result.model.year_created, etc.",
            style="blue",
        )
    )

    setup_observability(project_name="demo-pydantic-model")
    client = get_client()

    # ─── Single complete example ──────────────────────────────────────
    text = (
        "Python is a high-level programming language created by Guido van Rossum "
        "and first released in 1991. It follows a multi-paradigm approach, "
        "primarily object-oriented and imperative. Python uses dynamic typing "
        "with strong type enforcement at runtime. Popular frameworks include "
        "Django, Flask, FastAPI, and SQLAlchemy. The official website is "
        "python.org."
    )

    result: PydanticResult[ProgrammingLanguage] = pydantic_output(
        client,
        f"Extract programming language information as JSON from this text:\n{text}",
        ProgrammingLanguage,
        temperature=0.0,
        max_tokens=300,
    )

    # ─── Inspect the result ───────────────────────────────────────────
    console.print("\n[bold green]✅ Result:[/bold green]")

    if result.success and result.model:
        # Type-safe access to model fields
        model = result.model
        console.print(f"   [cyan]Name:[/cyan] {model.name}")
        console.print(f"   [cyan]Year:[/cyan] {model.year_created}")
        console.print(f"   [cyan]Creator:[/cyan] {model.creator}")
        console.print(f"   [cyan]Paradigm:[/cyan] {model.paradigm}")
        console.print(f"   [cyan]Typing:[/cyan] {model.typing}")
        console.print(
            f"   [cyan]Frameworks:[/cyan] {', '.join(model.popular_frameworks)}"
        )
        if model.website:
            console.print(f"   [cyan]Website:[/cyan] {model.website}")

        # Show full validated model as JSON
        console.print(f"\n   [dim]Full validated model:[/dim]")
        console.print_json(json.dumps(model.model_dump(), indent=2, default=str))
    else:
        console.print(f"   [red]Extraction failed[/red]")
        if result.validation_errors:
            for error in result.validation_errors:
                console.print(f"   [red]Validation error: {error}[/red]")

    if result.raw_result:
        console.print(
            f"\n   [dim]Duration: {result.raw_result.duration_ms:.0f}ms[/dim]"
        )
        if result.usage:
            console.print(
                f"   [dim]Tokens: {result.usage['prompt_tokens']} + "
                f"{result.usage['completion_tokens']} = "
                f"{result.usage['total_tokens']}[/dim]"
            )


if __name__ == "__main__":
    main()
