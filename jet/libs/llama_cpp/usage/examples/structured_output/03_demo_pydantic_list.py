"""Demo 03: List extraction using JSON Schema dict as response_format."""

from __future__ import annotations

import logging
from pathlib import Path

from jet.adapters.llama_cpp.factory import get_llm_client
from jet.libs.llama_cpp.usage.chat_stream_observability import (
    run_chat_stream,
    setup_observability,
)
from pydantic import BaseModel
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


class City(BaseModel):
    name: str
    country: str
    population_millions: float
    known_for: str | None = None


def main():
    console.print(
        Panel.fit(
            "📋 [bold]Demo 03: List Extraction via JSON Schema[/bold]\n"
            "Pass a JSON Schema dict to extract typed lists.",
            style="blue",
        )
    )
    setup_observability(project_name="demo-pydantic-list")
    client = get_llm_client()

    text = (
        "Tokyo, Japan has 37M people, known for technology. "
        "Delhi, India has 33M, famous for street food. "
        "Shanghai, China has 29M, global financial hub. "
        "São Paulo, Brazil has 22M, known for culture."
    )

    # Use JSON Schema dict for array extraction
    city_schema = {
        "type": "array",
        "items": City.model_json_schema(),
        "title": "CityList",
    }

    result = run_chat_stream(
        f"Extract ALL cities as a JSON array from:\n{text}",
        client=client,
        temperature=0.0,
        max_tokens=500,
        response_format=city_schema,
    )

    console.print("\n[bold green]✅ Result:[/bold green]")
    structured = getattr(result, "structured", None)
    if structured and structured.success and isinstance(structured.parsed, list):
        table = Table(title="Extracted Cities")
        table.add_column("City", style="cyan")
        table.add_column("Country", style="green")
        table.add_column("Pop (M)", justify="right")
        for item in structured.parsed:
            table.add_row(
                item.get("name", "?"),
                item.get("country", "?"),
                str(item.get("population_millions", "?")),
            )
        console.print(table)
        console.print(f"\n   [dim]Items: {len(structured.parsed)}[/dim]")
    else:
        console.print(f"   [red]Extraction failed[/red]")
        if structured:
            console.print(f"   [red]{structured.error}[/red]")


if __name__ == "__main__":
    main()
