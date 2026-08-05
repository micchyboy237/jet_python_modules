"""Demo: Stream structured JSON output from llama.cpp with Phoenix observability.

Uses response_format={"type": "json_object"} to guarantee valid JSON.
For stricter schema enforcement, pass extra_body={"grammar": "<GBNF>"} instead.

Demonstrates:
  1. Structured output via response_format parameter
  2. Programmatic access to content via StreamCompletionResult
  3. Full trace visibility in Phoenix with canonical /traces/ URL
  4. Resilient JSON extraction that handles markdown code fences
"""

from __future__ import annotations

import json
import logging
import re
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

# Regex to extract JSON from markdown code fences or raw text
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL)


def extract_json(raw: str) -> dict | list | None:
    """Extract and parse JSON from a response that may be wrapped in markdown fences.

    Smaller/local models frequently ignore response_format and wrap JSON
    in ```json ... ``` blocks. This helper handles both cases.
    """
    # Try direct parse first (ideal case: model obeyed json_object mode)
    stripped = raw.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code fence
    match = _JSON_FENCE_RE.search(stripped)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON inside code fence is invalid: {e}")
            return None

    # Try finding first { or [ and parsing from there
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = stripped.find(start_char)
        end = stripped.rfind(end_char)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(stripped[start : end + 1])
            except json.JSONDecodeError:
                continue

    return None


def main():
    setup_observability(project_name="structured-output-demo")
    client = get_client()

    prompt = (
        "Extract the key entities from this text and return them as JSON.\n\n"
        "Text: 'OpenTelemetry is an open-source observability framework created by CNCF. "
        "It provides APIs, SDKs, and tools for instrumenting, generating, collecting, "
        "and exporting telemetry data such as traces, metrics, and logs.'\n\n"
        "Return ONLY a JSON object with keys: 'framework_name', 'creator', "
        "'telemetry_types' (list). Do NOT wrap in markdown code fences."
    )

    logger.info("🏗️  Requesting structured JSON output...")
    result = run_chat_stream(
        client,
        prompt=prompt,
        model=MODEL,
        temperature=0.1,  # Lower temp for more reliable structured output
        response_format={"type": "json_object"},
    )

    # ── Use structured result ──────────────────────────────────────────
    logger.info(f"📋 Finish reason: {result.finish_reason}")
    if result.usage:
        logger.info(
            f"📊 Tokens: {result.usage['prompt_tokens']} prompt + "
            f"{result.usage['completion_tokens']} completion = "
            f"{result.usage['total_tokens']} total"
        )

    # ── Resilient JSON extraction ──────────────────────────────────────
    parsed = extract_json(result.content)
    if parsed is not None:
        console.print("\n[bold green]✅ Parsed JSON:[/bold green]")
        console.print_json(json.dumps(parsed, indent=2))
    else:
        logger.error(f"❌ Failed to extract JSON from response")
        logger.error(
            f"   Raw response ({len(result.content)} chars): {result.content[:300]}..."
        )


if __name__ == "__main__":
    main()
