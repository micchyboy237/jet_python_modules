"""Demo: Stream structured JSON output from llama.cpp with Phoenix observability.

Uses response_format={"type": "json_object"} to guarantee valid JSON.
For stricter schema enforcement, pass extra_body={"grammar": "<GBNF>"} instead.
"""

import json
import logging
from pathlib import Path

from jet.libs.llama_cpp.usage.chat_stream_vl_observability import (
    MODEL,
    get_client,
    run_chat_stream,
    setup_observability,
)
from rich.console import Console
from rich.logging import RichHandler

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, markup=True, rich_tracebacks=True)],
)
logger = logging.getLogger(Path(__file__).stem)


def main():
    setup_observability(project_name="structured-output-demo")
    client = get_client()

    prompt = (
        "Extract the key entities from this text and return them as JSON.\n\n"
        "Text: 'OpenTelemetry is an open-source observability framework created by CNCF. "
        "It provides APIs, SDKs, and tools for instrumenting, generating, collecting, "
        "and exporting telemetry data such as traces, metrics, and logs.'\n\n"
        "Return a JSON object with keys: 'framework_name', 'creator', 'telemetry_types' (list)."
    )

    logger.info("🏗️  Requesting structured JSON output...")
    raw_response = run_chat_stream(
        client,
        prompt=prompt,
        model=MODEL,
        temperature=0.1,  # Lower temp for more reliable structured output
        response_format={"type": "json_object"},
    )

    # Validate and pretty-print the JSON result
    try:
        parsed = json.loads(raw_response)
        console.print("\n[bold green]✅ Parsed JSON:[/bold green]")
        console.print_json(json.dumps(parsed, indent=2))
    except json.JSONDecodeError as e:
        logger.error(f"❌ Failed to parse JSON response: {e}")
        logger.error(f"   Raw response: {raw_response[:200]}...")


if __name__ == "__main__":
    main()
