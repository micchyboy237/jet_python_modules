"""Demo 04: OpenAI-compatible parsed completion pattern via run_chat_stream."""

from __future__ import annotations

import json
import logging
from pathlib import Path

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


class Conference(BaseModel):
    name: str = Field(description="Conference name")
    location: str = Field(description="City and country")
    year: int = Field(description="Year")
    attendees: int = Field(description="Approximate attendees")
    topics: list[str] = Field(description="Main topics")


def main():
    console.print(
        Panel.fit(
            "🔄 [bold]Demo 04: Parsed Completion Pattern[/bold]\n"
            "Pass Pydantic model → access result.structured.parsed",
            style="blue",
        )
    )
    setup_observability(project_name="demo-parsed-completion")
    client = get_client()

    text = (
        "PyCon US 2024 in Pittsburgh, PA attracted 2,500+ attendees. "
        "Topics: ML, web dev, DevOps, core Python."
    )

    result = run_chat_stream(
        f"Extract conference info from:\n{text}",
        client=client,
        temperature=0.0,
        max_tokens=300,
        response_format=Conference,
    )

    console.print("\n[bold green]✅ Result:[/bold green]")
    structured = getattr(result, "structured", None)
    if structured and structured.success and structured.parsed:
        conf = structured.parsed
        console.print(f"   [cyan]Name:[/cyan] {conf.name}")
        console.print(f"   [cyan]Location:[/cyan] {conf.location}")
        console.print(f"   [cyan]Attendees:[/cyan] {conf.attendees:,}")
        console.print(f"   [cyan]Topics:[/cyan] {', '.join(conf.topics)}")
        console.print(f"\n   [dim]Full model:[/dim]")
        console.print_json(json.dumps(conf.model_dump(), indent=2, default=str))
    else:
        console.print(f"   [red]Parsing failed[/red]")

    console.print(f"   [dim]Content: {len(result.content)} chars[/dim]")
    console.print(f"   [dim]Finish: {result.finish_reason}[/dim]")


if __name__ == "__main__":
    main()
