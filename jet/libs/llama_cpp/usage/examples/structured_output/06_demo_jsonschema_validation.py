"""Demo 06: Modern JSON Schema validation (Draft 2020-12) without Pydantic.

Demonstrates:
  1. Passing a raw JSON Schema dict (no Pydantic model)
  2. Automatic backend selection (encapsulated — client doesn't query it)
  3. Post-hoc validation via Draft202012Validator
  4. Modern schema features (prefixItems, contains, pattern, format)
  5. Detailed validation error reporting in StructuredResult
  6. Validator backend visible in logs + Phoenix trace attributes
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from jet.adapters.llama_cpp.factory import get_llm_client
from jet.libs.llama_cpp.usage.chat_stream_observability import (
    run_chat_stream,
    setup_observability,
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

SENSOR_READING_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "SensorReading",
    "type": "object",
    "properties": {
        "sensor_id": {
            "type": "string",
            "pattern": "^SNS-[0-9]{4}$",
            "description": "Sensor identifier, format SNS-XXXX",
        },
        "timestamp": {
            "type": "string",
            "format": "date-time",
            "description": "ISO 8601 timestamp",
        },
        "readings": {
            "type": "array",
            "prefixItems": [
                {
                    "description": "First reading: temperature (°C)",
                    "type": "number",
                    "minimum": -50,
                    "maximum": 150,
                },
                {
                    "description": "Second reading: humidity (%)",
                    "type": "number",
                    "minimum": 0,
                    "maximum": 100,
                },
            ],
            "items": {"type": "number"},
            "minItems": 2,
            "maxItems": 5,
            "description": "Array where first two elements are temp then humidity",
        },
        "status": {
            "type": "string",
            "enum": ["nominal", "warning", "critical"],
            "description": "Current sensor status",
        },
        "tags": {
            "type": "array",
            "contains": {"const": "production"},
            "items": {"type": "string"},
            "description": "Tags array; must include 'production'",
        },
    },
    "required": ["sensor_id", "timestamp", "readings", "status", "tags"],
    "additionalProperties": False,
}


def main():
    console.print(
        Panel.fit(
            "🔍 [bold]Demo 06: Modern JSON Schema Validation[/bold]\n"
            "Raw schema dict + Draft 2020-12 validator\n"
            "Backend auto-selected & logged (no client code needed)\n"
            "(No Pydantic model involved)",
            style="blue",
        )
    )

    setup_observability(project_name="demo-jsonschema-modern")
    client = get_llm_client()

    prompt = (
        "Generate a sensor reading for an industrial IoT device.\n"
        "Sensor ID format: SNS-XXXX (4 digits).\n"
        "Include timestamp (ISO 8601), readings array (temp then humidity first),\n"
        "status (nominal/warning/critical), and tags (must include 'production').\n"
        "Return ONLY valid JSON matching the schema."
    )

    result = run_chat_stream(
        prompt,
        client=client,
        temperature=0.0,
        max_tokens=400,
        response_format=SENSOR_READING_SCHEMA,
    )

    console.print("\n[bold green]✅ Raw Response:[/bold green]")
    console.print(f"   [dim]{result.content[:300]}[/dim]")

    structured = getattr(result, "structured", None)
    if structured is None:
        console.print("[red]No structured result attached[/red]")
        return

    if structured.success:
        console.print("\n[bold cyan]✅ Schema Validation PASSED[/bold cyan]")
        console.print_json(json.dumps(structured.parsed, indent=2, default=str))
    else:
        console.print(f"\n[bold red]❌ Schema Validation FAILED[/bold red]")
        console.print(f"   Error: {structured.error}")
        if structured.validation_errors:
            console.print("\n   [yellow]Validation details:[/yellow]")
            for i, err in enumerate(structured.validation_errors, 1):
                console.print(f"   {i}. {err}")
        if structured.parsed:
            console.print("\n   [dim]Extracted (invalid) JSON:[/dim]")
            console.print_json(json.dumps(structured.parsed, indent=2, default=str))

    console.print(f"\n   [dim]Finish reason: {result.finish_reason}[/dim]")
    console.print(
        "[dim]💡 Validator backend visible in logs above and Phoenix trace attributes[/dim]"
    )


if __name__ == "__main__":
    main()
