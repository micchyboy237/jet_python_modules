# jet_python_modules/jet/libs/llama_cpp/usage/examples/chat_stream/06_demo_response_formats.py
"""Demo: All OpenAI response formats compatible with llama.cpp.
Demonstrates:
  1. text mode (default)
  2. json_object mode
  3. grammar-constrained mode (GBNF)
  4. function calling
  5. Comparison table with success rates

llama.cpp response_format support summary:
  ✅ {"type": "text"}           - Works (default)
  ✅ {"type": "json_object"}    - Works (best-effort, not guaranteed)
  ⚠️ {"type": "json_schema"}    - Schema ignored, falls back to json_object
  ❌ custom_tool format=grammar  - Not supported via response_format
  ✅ grammar via extra_body     - Works perfectly (GBNF grammar)
  ✅ function calling via tools - Works with proper tool definitions
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from jet.libs.llama_cpp.usage.chat_stream_observability import (
    MODEL,
    get_client,
    run_chat_stream,
    setup_observability,
)
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


class FormatType(Enum):
    """Response format types compatible with llama.cpp."""

    TEXT = "text"
    JSON_OBJECT = "json_object"
    GRAMMAR = "grammar"
    FUNCTION_CALLING = "function_calling"


@dataclass
class FormatTestResult:
    """Single test result for a response format."""

    format_type: FormatType
    success: bool
    duration_ms: float
    raw_response: str
    parsed_json: dict | list | None = None
    error: str | None = None
    token_count: int = 0
    notes: str = ""


@dataclass
class FormatComparison:
    """Complete comparison of all format tests."""

    results: list[FormatTestResult] = field(default_factory=list)

    @property
    def success_count(self) -> int:
        return sum(1 for r in self.results if r.success)

    @property
    def total_count(self) -> int:
        return len(self.results)

    def print_table(self):
        """Display a formatted comparison table."""
        table = Table(title="Response Format Comparison - llama.cpp Compatibility")
        table.add_column("Format", style="cyan")
        table.add_column("Status", style="bold")
        table.add_column("Time (ms)", justify="right")
        table.add_column("Tokens", justify="right")
        table.add_column("Notes", style="dim")

        for result in self.results:
            status = "✅ PASS" if result.success else "❌ FAIL"
            status_style = "green" if result.success else "red"

            table.add_row(
                result.format_type.value,
                f"[{status_style}]{status}[/{status_style}]",
                f"{result.duration_ms:.0f}",
                str(result.token_count),
                result.notes or "-",
            )

        summary = (
            f"Success rate: {self.success_count}/{self.total_count} "
            f"({self.success_count / self.total_count * 100:.0f}%)"
        )
        console.print(table)
        console.print(Panel(summary, style="bold blue"))


# ─── GBNF Grammar definitions ───────────────────────────────────────────────

GBNF_SIMPLE_JSON = r"""
root   ::= object
object ::= "{" ws string ws ":" ws string ws "}" 
string ::= "\"" [a-zA-Z0-9\s\.\,\!\?\-\+\/\\_@#\$%^&*\(\)\[\]\{\}\|\:\;\<\>\~\`\']* "\""
ws     ::= [ \t\n]*
"""

GBNF_PERSON_JSON = r"""
root   ::= person
person ::= "{" ws "\"name\"" ws ":" ws string ws "," ws "\"age\"" ws ":" ws number ws "," ws "\"city\"" ws ":" ws string ws "}"
string ::= "\"" [a-zA-Z\s]* "\""
number ::= [0-9]+
ws     ::= [ \t\n]*
"""

GBNF_WEATHER = r"""
root   ::= object
object ::= "{" ws "\"location\"" ws ":" ws string ws "," ws "\"temperature\"" ws ":" ws number ws "," ws "\"condition\"" ws ":" ws string ws "}"
string ::= "\"" [a-zA-Z\s]* "\""
number ::= [0-9]+(\.[0-9]+)?
ws     ::= [ \t\n]*
"""


def extract_json(raw: str) -> dict | list | None:
    """Extract JSON from response that may have markdown fences."""
    stripped = raw.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # Try markdown code fences
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", stripped, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try finding first { or [ pair
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = stripped.find(start_char)
        end = stripped.rfind(end_char)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(stripped[start : end + 1])
            except json.JSONDecodeError:
                continue
    return None


def test_text_format(client) -> FormatTestResult:
    """Test 1: Plain text response (always works)."""
    prompt = "Say 'Hello, World!' and nothing else."

    t0 = time.perf_counter()
    result = run_chat_stream(
        client,
        prompt=prompt,
        model=MODEL,
        temperature=0.0,
        max_tokens=50,
    )
    duration_ms = (time.perf_counter() - t0) * 1000

    token_count = result.usage.get("completion_tokens", 0) if result.usage else 0

    return FormatTestResult(
        format_type=FormatType.TEXT,
        success=True,  # Text mode always works
        duration_ms=duration_ms,
        raw_response=result.content,
        token_count=token_count,
        notes="Default mode, always works",
    )


def test_json_object_format(client) -> FormatTestResult:
    """Test 2: JSON object mode via response_format."""
    prompt = (
        "Return a JSON object with fields: 'name' (string), 'age' (number), "
        "'city' (string). Use these values: name='Alice', age=30, city='Paris'.\n"
        "IMPORTANT: Return ONLY the JSON object, no markdown, no explanation."
    )

    t0 = time.perf_counter()
    result = run_chat_stream(
        client,
        prompt=prompt,
        model=MODEL,
        temperature=0.0,
        max_tokens=200,
        response_format={"type": "json_object"},
    )
    duration_ms = (time.perf_counter() - t0) * 1000

    token_count = result.usage.get("completion_tokens", 0) if result.usage else 0
    parsed = extract_json(result.content)

    success = parsed is not None and isinstance(parsed, dict)
    error = None
    if not success:
        error = "Failed to parse JSON from response"

    return FormatTestResult(
        format_type=FormatType.JSON_OBJECT,
        success=success,
        duration_ms=duration_ms,
        raw_response=result.content,
        parsed_json=parsed,
        token_count=token_count,
        error=error,
        notes="Best-effort JSON, may wrap in ``` fences",
    )


def test_grammar_format(client) -> FormatTestResult:
    """Test 3: Grammar-constrained output via extra_body."""
    prompt = "Generate a person profile as JSON."

    # Note: Grammar is passed via extra_body, NOT response_format
    t0 = time.perf_counter()
    # We need to call run_chat_stream with extra_body for grammar
    # Since run_chat_stream doesn't directly expose extra_body,
    # we'll use the client directly for this test
    try:
        stream = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.0,
            stream=True,
            stream_options={"include_usage": True},
            extra_body={
                "grammar": GBNF_PERSON_JSON,
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )

        collected = []
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                collected.append(chunk.choices[0].delta.content)

        raw = "".join(collected)
        duration_ms = (time.perf_counter() - t0) * 1000
        parsed = extract_json(raw)
        success = parsed is not None and isinstance(parsed, dict)

        return FormatTestResult(
            format_type=FormatType.GRAMMAR,
            success=success,
            duration_ms=duration_ms,
            raw_response=raw,
            parsed_json=parsed,
            error=None if success else "Grammar output not valid JSON",
            notes="GBNF grammar via extra_body",
        )
    except Exception as e:
        duration_ms = (time.perf_counter() - t0) * 1000
        return FormatTestResult(
            format_type=FormatType.GRAMMAR,
            success=False,
            duration_ms=duration_ms,
            raw_response="",
            error=str(e),
            notes=f"Grammar not supported by this model",
        )


def test_function_calling_format(client) -> FormatTestResult:
    """Test 4: Function calling for structured output."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_person_info",
                "description": "Return information about a person",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                        "city": {"type": "string"},
                    },
                    "required": ["name", "age", "city"],
                },
            },
        }
    ]

    prompt = "Tell me about a person named Bob who is 25 and lives in London."

    t0 = time.perf_counter()
    result = run_chat_stream(
        client,
        prompt=prompt,
        model=MODEL,
        temperature=0.0,
        max_tokens=200,
        tools=tools,
        tool_choice="auto",
    )
    duration_ms = (time.perf_counter() - t0) * 1000

    token_count = result.usage.get("completion_tokens", 0) if result.usage else 0
    success = result.has_tool_calls

    parsed_args = None
    if success and result.tool_calls:
        parsed_args = result.tool_calls[0].arguments

    return FormatTestResult(
        format_type=FormatType.FUNCTION_CALLING,
        success=success,
        duration_ms=duration_ms,
        raw_response=result.content,
        parsed_json=parsed_args,
        token_count=token_count,
        notes="Structured via function arguments"
        if success
        else "Model didn't call tool",
    )


def main():
    """Run all format tests and compare results."""
    console.print(
        Panel.fit(
            "🧪 [bold]OpenAI Response Format Compatibility Tests[/bold]\n"
            "Testing which formats work with llama.cpp server",
            style="blue",
        )
    )

    setup_observability(project_name="response-formats-demo")
    client = get_client()

    comparison = FormatComparison()

    # Test 1: Text (always works)
    console.print("\n[bold yellow]Test 1/4: Text format[/bold yellow]")
    result = test_text_format(client)
    comparison.results.append(result)
    console.print(f"  {'✅' if result.success else '❌'} {result.notes}")

    # Test 2: JSON Object
    console.print("\n[bold yellow]Test 2/4: JSON Object format[/bold yellow]")
    result = test_json_object_format(client)
    comparison.results.append(result)
    console.print(f"  {'✅' if result.success else '❌'} {result.notes}")
    if result.error:
        console.print(f"  [red]Error: {result.error}[/red]")
    if result.parsed_json:
        console.print(f"  [green]Parsed: {json.dumps(result.parsed_json)}[/green]")

    # Test 3: Grammar
    console.print("\n[bold yellow]Test 3/4: Grammar (GBNF) format[/bold yellow]")
    result = test_grammar_format(client)
    comparison.results.append(result)
    console.print(f"  {'✅' if result.success else '❌'} {result.notes}")
    if result.error:
        console.print(f"  [red]Error: {result.error}[/red]")
    if result.parsed_json:
        console.print(f"  [green]Parsed: {json.dumps(result.parsed_json)}[/green]")

    # Test 4: Function Calling
    console.print("\n[bold yellow]Test 4/4: Function Calling format[/bold yellow]")
    result = test_function_calling_format(client)
    comparison.results.append(result)
    console.print(f"  {'✅' if result.success else '❌'} {result.notes}")
    if result.parsed_json:
        console.print(f"  [green]Arguments: {json.dumps(result.parsed_json)}[/green]")

    # Summary
    console.print("\n")
    comparison.print_table()

    # Recommendations
    console.print(
        Panel(
            "[bold]Recommendations for llama.cpp:[/bold]\n\n"
            "1. 🥇 [green]Grammar (GBNF)[/green] - Most reliable for structured output\n"
            "2. 🥈 [green]Function Calling[/green] - Great for tool-based extraction\n"
            "3. 🥉 [yellow]JSON Object[/yellow] - Works but requires JSON extraction\n"
            "4. ❌ [red]JSON Schema[/red] - Schema constraints are ignored\n"
            "5. ✅ [green]Text[/green] - Always works, use with regex parsing\n\n"
            "[dim]Tip: For production use, combine grammar + function calling "
            "for the most reliable structured output.[/dim]",
            style="blue",
            title="📊 Summary",
        )
    )


if __name__ == "__main__":
    main()
