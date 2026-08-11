"""Demo 03: List of Pydantic models from a single prompt.

Demonstrates extracting multiple entities into a typed list.
Useful for batch extraction from paragraphs containing multiple items.

What this shows:
  - pydantic_list_output() for extracting lists
  - Iterating over validated model instances
  - Handling partial successes (some items validate, some don't)
  - How to access: result.model (list of Pydantic instances)
  - Type-safe iteration: for item in result.model: print(item.name)
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
    pydantic_list_output,
)
from pydantic import BaseModel, Field
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


# ─── Define the Pydantic model once ────────────────────────────────────────


class City(BaseModel):
    """Information about a city."""

    name: str = Field(description="City name")
    country: str = Field(description="Country name")
    population_millions: float = Field(description="Population in millions")
    known_for: Optional[str] = Field(
        default=None, description="What the city is famous for"
    )


def main():
    console.print(
        Panel.fit(
            "📋 [bold]Demo 03: Pydantic List Extraction[/bold]\n"
            "Extract multiple entities into a typed list.\n"
            "Iterate with: for city in result.model: print(city.name)",
            style="blue",
        )
    )

    setup_observability(project_name="demo-pydantic-list")
    client = get_client()

    # ─── Single complete example ──────────────────────────────────────
    text = (
        "Tokyo, Japan has a population of about 37 million and is known for "
        "its technology and cuisine. Delhi, India has around 33 million people "
        "and is famous for its history and street food. Shanghai, China has "
        "approximately 29 million residents and is a global financial hub. "
        "São Paulo, Brazil has about 22 million people and is known for its "
        "culture and architecture."
    )

    result: PydanticResult[list[City]] = pydantic_list_output(
        client=client,
        f"Extract ALL cities mentioned as a JSON array from this text:\n{text}",
        City,
        temperature=0.0,
        max_tokens=500,
    )

    # ─── Inspect the result ───────────────────────────────────────────
    console.print("\n[bold green]✅ Result:[/bold green]")

    if result.success and result.model:
        # Display as a table
        table = Table(title="Extracted Cities")
        table.add_column("City", style="cyan")
        table.add_column("Country", style="green")
        table.add_column("Population (M)", justify="right")
        table.add_column("Known For", style="yellow")

        for city in result.model:
            table.add_row(
                city.name,
                city.country,
                str(city.population_millions),
                city.known_for or "-",
            )

        console.print(table)
        console.print(f"\n   [dim]Validated: {len(result.model)} cities[/dim]")

        # Show raw JSON
        console.print(f"\n   [dim]As JSON:[/dim]")
        raw_json = [c.model_dump() for c in result.model]
        console.print_json(json.dumps(raw_json, indent=2, default=str))
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
