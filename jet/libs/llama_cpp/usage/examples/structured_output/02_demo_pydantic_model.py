"""Demo 02: Pydantic model passed directly as response_format."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from jet.libs.llama_cpp.usage.chat_stream_observability import (
    get_client,
    run_chat_stream,
    setup_observability,
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


class ProgrammingLanguage(BaseModel):
    name: str = Field(description="Name of the language")
    year_created: int = Field(description="Year first released")
    creator: str = Field(description="Creator or organization")
    paradigm: str = Field(description="Programming paradigm")
    typing: str = Field(description="Type system")
    popular_frameworks: list[str] = Field(description="Popular frameworks")
    website: Optional[str] = Field(default=None, description="Official website")


def main():
    console.print(
        Panel.fit(
            "🔒 [bold]Demo 02: Pydantic Model as response_format[/bold]\n"
            "Pass the model class directly — schema & validation handled automatically.",
            style="blue",
        )
    )
    setup_observability(project_name="demo-pydantic-model")
    client = get_client()

    text = (
        "Python is a high-level language created by Guido van Rossum in 1991. "
        "Multi-paradigm, primarily OOP. Dynamic typing with strong enforcement. "
        "Frameworks: Django, Flask, FastAPI, SQLAlchemy. Website: python.org."
    )

    # ← Pass Pydantic model DIRECTLY as response_format
    result = run_chat_stream(
        f"Extract programming language info from:\n{text}",
        client=client,
        temperature=0.0,
        max_tokens=300,
        response_format=ProgrammingLanguage,
    )

    console.print("\n[bold green]✅ Result:[/bold green]")
    structured = getattr(result, "structured", None)
    if structured and structured.success and structured.parsed:
        model = structured.parsed
        console.print(f"   [cyan]Name:[/cyan] {model.name}")
        console.print(f"   [cyan]Year:[/cyan] {model.year_created}")
        console.print(f"   [cyan]Creator:[/cyan] {model.creator}")
        console.print(
            f"   [cyan]Frameworks:[/cyan] {', '.join(model.popular_frameworks)}"
        )
        console.print(f"\n   [dim]Full model:[/dim]")
        console.print_json(json.dumps(model.model_dump(), indent=2, default=str))
    else:
        console.print(f"   [red]Extraction failed[/red]")
        if structured:
            for err in structured.validation_errors:
                console.print(f"   [red]{err}[/red]")


if __name__ == "__main__":
    main()
