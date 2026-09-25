"""Demo 01: JSON object output via response_format."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from jet.adapters.llama_cpp.factory import get_llm_client
from jet.libs.llama_cpp.usage.chat_stream_observability import (
    run_chat_stream,
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

# Setup output directory
OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    console.print(
        Panel.fit(
            "🏗️ [bold]Demo 01: JSON Object Output[/bold]\n"
            "Best-effort JSON via response_format={'type': 'json_object'}",
            style="blue",
        )
    )
    client = get_llm_client()
    prompt = (
        "Extract key facts about Python as a JSON object with fields:\n"
        '  "creator" (string), "year_created" (number), "type" (string),\n'
        '  "known_for" (array), "latest_version" (string)\n'
        "Python was created by Guido van Rossum in 1991. "
        "It is an interpreted, high-level language. Latest version is 3.13.\n"
        "Return ONLY JSON."
    )

    result = run_chat_stream(
        prompt,
        client=client,
        project_name="demo-json-object",
        temperature=0.0,
        max_tokens=300,
        response_format={"type": "json_object"},
        output_dir=OUTPUT_DIR,
    )

    console.print("\n[bold green]✅ Result:[/bold green]")
    console.print(f"   [dim]Raw: {result.content[:200]}[/dim]")
    structured = getattr(result, "structured", None)
    if structured and structured.success:
        console.print(f"\n[cyan]Parsed JSON:[/cyan]")
        console.print_json(json.dumps(structured.parsed, indent=2))
    else:
        console.print(f"\n[yellow]No structured parse available[/yellow]")
    console.print(f"   [dim]Finish: {result.finish_reason}[/dim]")


if __name__ == "__main__":
    main()
