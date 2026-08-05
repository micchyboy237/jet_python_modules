# jet_python_modules/jet/libs/llama_cpp/structured_output.py
"""Structured output helpers for llama.cpp OpenAI-compatible server.

Provides encapsulated, reusable functions for each response format
that actually works with llama.cpp:

  - text_output()          → Plain text (always works)
  - json_object_output()   → Best-effort JSON via response_format
  - grammar_output()       → Strict JSON via GBNF grammar
  - function_call_output() → Structured output via function calling
  - auto_structured()      → Smart auto-selection based on requirements

Design principles:
  - Each function returns a typed dataclass, not raw dict/str
  - Built-in JSON extraction handles common llama.cpp quirks
  - Grammar definitions are pre-tested and versioned
  - Comprehensive logging at every step
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from jet.libs.llama_cpp.usage.chat_stream_observability import (
    MODEL as DEFAULT_MODEL,
)
from jet.libs.llama_cpp.usage.chat_stream_observability import (
    run_chat_stream,
)
from openai import OpenAI

logger = logging.getLogger(Path(__file__).stem)


# ─── Data Classes ──────────────────────────────────────────────────────────


class OutputFormat(Enum):
    """Supported output formats for llama.cpp."""

    TEXT = "text"
    JSON_OBJECT = "json_object"
    GRAMMAR = "grammar"
    FUNCTION_CALL = "function_call"


@dataclass
class StructuredResult:
    """Unified result from any structured output method.

    Attributes:
        format_used: Which format produced this result
        success: Whether valid structured output was obtained
        content: Raw text content from the model
        parsed: Parsed JSON (if applicable)
        tool_calls: Parsed tool calls (if applicable)
        usage: Token usage stats
        finish_reason: Why model stopped generating
        duration_ms: Total round-trip time in milliseconds
        error: Error message if failed
    """

    format_used: OutputFormat
    success: bool
    content: str
    parsed: dict | list | None = None
    tool_calls: list[dict[str, Any]] | None = None
    usage: dict[str, int] | None = None
    finish_reason: str | None = None
    duration_ms: float = 0.0
    error: str | None = None

    @property
    def as_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "format_used": self.format_used.value,
            "success": self.success,
            "content": self.content,
            "parsed": self.parsed,
            "tool_calls": self.tool_calls,
            "usage": self.usage,
            "finish_reason": self.finish_reason,
            "duration_ms": self.duration_ms,
            "error": self.error,
        }


# ─── JSON Extraction Utilities ─────────────────────────────────────────────

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"(\{.*\})", re.DOTALL)
_JSON_ARRAY_RE = re.compile(r"(\[.*\])", re.DOTALL)


def extract_json(raw: str) -> dict | list | None:
    """Robustly extract JSON from model output.

    Handles common llama.cpp output quirks:
    - Direct JSON: {"key": "value"}
    - Markdown fences: ```json\n{...}\n```
    - Plain fences: ```\n{...}\n```
    - Embedded JSON in text: "Here is...\n{...}\nHope this helps"

    Args:
        raw: Raw string output from the model

    Returns:
        Parsed JSON dict/list, or None if extraction fails
    """
    stripped = raw.strip()

    # Attempt 1: Direct parse
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # Attempt 2: Extract from markdown code fences
    match = _JSON_FENCE_RE.search(stripped)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Attempt 3: Find last complete JSON object
    for pattern in [_JSON_OBJECT_RE, _JSON_ARRAY_RE]:
        matches = pattern.findall(stripped)
        for candidate in reversed(matches):  # Last one is usually correct
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, (dict, list)):
                    return parsed
            except json.JSONDecodeError:
                continue

    return None


# ─── Pre-built Grammar Templates ──────────────────────────────────────────


@dataclass
class GrammarTemplate:
    """A reusable GBNF grammar template."""

    name: str
    grammar: str
    description: str = ""


GRAMMAR_TEMPLATES = {
    "simple_object": GrammarTemplate(
        name="simple_object",
        grammar=r"""
root   ::= object
object ::= "{" ws string ws ":" ws string ws "}" 
string ::= "\"" [a-zA-Z0-9\s\.\,\!\?\-\+\/\\_@#\$%^&*\(\)\[\]\{\}\|\:\;\<\>\~\`\']* "\""
ws     ::= [ \t\n]*
""",
        description="A JSON object with a single string key-value pair",
    ),
    "person": GrammarTemplate(
        name="person",
        grammar=r"""
root   ::= person
person ::= "{" ws "\"name\"" ws ":" ws string ws "," ws "\"age\"" ws ":" ws number ws "," ws "\"city\"" ws ":" ws string ws "}"
string ::= "\"" [a-zA-Z\s]* "\""
number ::= [0-9]+
ws     ::= [ \t\n]*
""",
        description="Person object with name (string), age (number), city (string)",
    ),
    "weather": GrammarTemplate(
        name="weather",
        grammar=r"""
root   ::= object
object ::= "{" ws "\"location\"" ws ":" ws string ws "," ws "\"temperature\"" ws ":" ws number ws "," ws "\"condition\"" ws ":" ws string ws "}"
string ::= "\"" [a-zA-Z\s]* "\""
number ::= [0-9]+(\.[0-9]+)?
ws     ::= [ \t\n]*
""",
        description="Weather object with location, temperature, condition",
    ),
    "list_of_strings": GrammarTemplate(
        name="list_of_strings",
        grammar=r"""
root   ::= array
array  ::= "[" ws string (ws "," ws string)* ws "]"
string ::= "\"" [a-zA-Z0-9\s\.\,\!\?\-\+\/]* "\""
ws     ::= [ \t\n]*
""",
        description="JSON array of strings",
    ),
}


def grammar_from_json_schema(schema: dict[str, Any]) -> str:
    """Convert a JSON Schema to GBNF grammar (simplified converter).

    This is a basic converter for common patterns. For complex schemas,
    use a dedicated converter library.

    Supported types: string, integer, number, boolean, object (shallow), array

    Args:
        schema: JSON Schema dict

    Returns:
        GBNF grammar string
    """
    props = schema.get("properties", {})
    required = schema.get("required", [])

    if not props:
        # Fallback: allow any JSON
        return r"""
root   ::= object
object ::= "{" ws ( string ws ":" ws value ws ("," ws string ws ":" ws value ws)* )? ws "}"
string ::= "\"" [a-zA-Z0-9\s\.\,\!\?\-\+\/\\_@#\$%^&*\(\)\[\]\{\}\|\:\;\<\>\~\`]* "\""
value  ::= string | number | boolean | object | array
number ::= "-"? [0-9]+ ("." [0-9]+)?
boolean ::= "true" | "false"
array  ::= "[" ws (value (ws "," ws value)*)? ws "]"
ws     ::= [ \t\n]*
"""

    # Build field definitions
    field_defs = []
    for name, prop in props.items():
        ptype = prop.get("type", "string")
        if ptype == "string":
            field_defs.append(f'"{name}" ws ":" ws string')
        elif ptype in ("integer", "number"):
            field_defs.append(f'"{name}" ws ":" ws number')
        elif ptype == "boolean":
            field_defs.append(f'"{name}" ws ":" ws boolean')
        elif ptype == "array":
            field_defs.append(f'"{name}" ws ":" ws array')
        elif ptype == "object":
            field_defs.append(f'"{name}" ws ":" ws object')

    fields_grammar = " ws ", " ws ".join(field_defs)

    return f"""
root   ::= object
object ::= "{{" ws {fields_grammar} ws "}}"
string ::= "\\"" [a-zA-Z0-9\\s\\.\\,\\!\\?\\-\\+\\/\\\\_@#\\$%^&*\\(\\)\\[\\]\\{{\\}}\\|\\:\\;\\<\\>\\~\\`]* "\\""
number ::= "-"? [0-9]+ ("." [0-9]+)?
boolean ::= "true" | "false"
array  ::= "[" ws (value (ws "," ws value)*)? ws "]"
value  ::= string | number | boolean | object | array
ws     ::= [ \\t\\n]*
"""


# ─── Core Output Functions ────────────────────────────────────────────────


def text_output(
    client: OpenAI,
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    **kwargs: Any,
) -> StructuredResult:
    """Get plain text output (always works, no structure).

    Args:
        client: OpenAI client pointing to llama.cpp server
        prompt: User prompt
        model: Model name
        temperature: Sampling temperature (0.0 for deterministic)
        max_tokens: Max output tokens
        **kwargs: Passed to run_chat_stream

    Returns:
        StructuredResult with format_used=TEXT, success=True
    """
    t0 = time.perf_counter()

    logger.debug(f"📝 [text_output] Starting with prompt: {prompt[:80]}...")

    result = run_chat_stream(
        client,
        prompt=prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )

    duration_ms = (time.perf_counter() - t0) * 1000

    logger.debug(
        f"✅ [text_output] Complete in {duration_ms:.0f}ms, "
        f"{len(result.content)} chars, finish={result.finish_reason}"
    )

    return StructuredResult(
        format_used=OutputFormat.TEXT,
        success=True,
        content=result.content,
        usage=result.usage,
        finish_reason=result.finish_reason,
        duration_ms=duration_ms,
    )


def json_object_output(
    client: OpenAI,
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    extractor: Callable[[str], dict | list | None] = extract_json,
    **kwargs: Any,
) -> StructuredResult:
    """Get JSON output using response_format={"type": "json_object"}.

    Note: llama.cpp treats this as a best-effort hint. The model may
    still output non-JSON content or wrap JSON in markdown fences.
    This function handles those cases automatically.

    Args:
        client: OpenAI client
        prompt: User prompt (should include "Return JSON" instruction)
        model: Model name
        temperature: Low temperature recommended (0.0-0.1)
        max_tokens: Max output tokens
        extractor: Custom JSON extraction function (default: extract_json)
        **kwargs: Passed to run_chat_stream

    Returns:
        StructuredResult with parsed JSON if successful
    """
    t0 = time.perf_counter()

    logger.debug(f"🏗️ [json_object_output] Starting with prompt: {prompt[:80]}...")

    result = run_chat_stream(
        client,
        prompt=prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
        **kwargs,
    )

    duration_ms = (time.perf_counter() - t0) * 1000
    parsed = extractor(result.content)
    success = parsed is not None

    if success:
        logger.debug(f"✅ [json_object_output] Parsed JSON in {duration_ms:.0f}ms")
    else:
        logger.warning(
            f"⚠️ [json_object_output] Failed to parse JSON. "
            f"Raw: {result.content[:100]}..."
        )

    return StructuredResult(
        format_used=OutputFormat.JSON_OBJECT,
        success=success,
        content=result.content,
        parsed=parsed,
        usage=result.usage,
        finish_reason=result.finish_reason,
        duration_ms=duration_ms,
        error=None if success else "Failed to extract valid JSON from response",
    )


def grammar_output(
    client: OpenAI,
    prompt: str,
    grammar: str,
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    grammar_name: str = "custom",
    extractor: Callable[[str], dict | list | None] = extract_json,
    **kwargs: Any,
) -> StructuredResult:
    """Get strictly structured output using GBNF grammar.

    This is the MOST RELIABLE method for structured output with llama.cpp.
    The grammar constrains token generation to produce valid syntax.

    Args:
        client: OpenAI client
        prompt: User prompt
        grammar: GBNF grammar string (use GRAMMAR_TEMPLATES or grammar_from_json_schema)
        model: Model name
        temperature: Use 0.0 for deterministic output
        max_tokens: Max output tokens
        grammar_name: Label for logging
        extractor: JSON extraction function
        **kwargs: Passed to chat.completions.create

    Returns:
        StructuredResult with guaranteed valid JSON syntax
    """
    t0 = time.perf_counter()

    logger.debug(
        f"🔒 [grammar_output] Using grammar '{grammar_name}', prompt: {prompt[:80]}..."
    )

    try:
        # Grammar must be passed via extra_body, not response_format
        extra = kwargs.pop("extra_body", {})
        extra["grammar"] = grammar
        extra.setdefault("top_k", 20)
        extra.setdefault("chat_template_kwargs", {"enable_thinking": False})

        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            stream_options={"include_usage": True},
            extra_body=extra,
        )

        collected: list[str] = []
        usage = None
        finish_reason = None

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                collected.append(chunk.choices[0].delta.content)
            if chunk.choices and chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason
            if hasattr(chunk, "usage") and chunk.usage:
                usage = {
                    "prompt_tokens": chunk.usage.prompt_tokens,
                    "completion_tokens": chunk.usage.completion_tokens,
                    "total_tokens": chunk.usage.total_tokens,
                }

        content = "".join(collected)
        duration_ms = (time.perf_counter() - t0) * 1000
        parsed = extractor(content)
        success = parsed is not None

        logger.debug(
            f"{'✅' if success else '⚠️'} [grammar_output] "
            f"Complete in {duration_ms:.0f}ms, "
            f"parsed={'ok' if success else 'FAIL'}"
        )

        return StructuredResult(
            format_used=OutputFormat.GRAMMAR,
            success=success,
            content=content,
            parsed=parsed,
            usage=usage,
            finish_reason=finish_reason,
            duration_ms=duration_ms,
            error=None if success else "Grammar output could not be parsed as JSON",
        )

    except Exception as e:
        duration_ms = (time.perf_counter() - t0) * 1000
        logger.error(f"❌ [grammar_output] Failed: {e}")

        return StructuredResult(
            format_used=OutputFormat.GRAMMAR,
            success=False,
            content="",
            usage=None,
            finish_reason=None,
            duration_ms=duration_ms,
            error=str(e),
        )


def function_call_output(
    client: OpenAI,
    prompt: str,
    tools: list[dict[str, Any]],
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    tool_choice: str | dict[str, Any] = "auto",
    **kwargs: Any,
) -> StructuredResult:
    """Get structured output via function calling.

    The model is given function definitions and can choose to call them.
    This provides structured argument extraction in a validated format.

    Args:
        client: OpenAI client
        prompt: User prompt
        tools: List of tool definitions (OpenAI format)
        model: Model name
        temperature: Low temperature for reliable tool calling
        max_tokens: Max output tokens
        tool_choice: "auto", "required", "none", or specific tool dict
        **kwargs: Passed to run_chat_stream

    Returns:
        StructuredResult with tool_calls containing parsed arguments
    """
    t0 = time.perf_counter()

    tool_names = [t.get("function", {}).get("name", "?") for t in tools]
    logger.debug(
        f"🔧 [function_call_output] Tools: {tool_names}, prompt: {prompt[:80]}..."
    )

    result = run_chat_stream(
        client,
        prompt=prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        tools=tools,
        tool_choice=tool_choice,
        **kwargs,
    )

    duration_ms = (time.perf_counter() - t0) * 1000
    success = result.has_tool_calls

    tool_calls_dict = None
    if success:
        tool_calls_dict = [
            {
                "id": tc.id,
                "name": tc.name,
                "arguments": tc.arguments,
            }
            for tc in result.tool_calls
        ]
        logger.debug(
            f"✅ [function_call_output] {len(result.tool_calls)} tool call(s) "
            f"in {duration_ms:.0f}ms"
        )
    else:
        logger.warning(
            f"⚠️ [function_call_output] No tool calls generated. "
            f"Content: {result.content[:100]}..."
        )

    return StructuredResult(
        format_used=OutputFormat.FUNCTION_CALL,
        success=success,
        content=result.content,
        parsed=result.tool_calls[0].arguments if result.tool_calls else None,
        tool_calls=tool_calls_dict,
        usage=result.usage,
        finish_reason=result.finish_reason,
        duration_ms=duration_ms,
        error=None if success else "Model did not call any tool",
    )


def auto_structured(
    client: OpenAI,
    prompt: str,
    *,
    json_schema: dict[str, Any] | None = None,
    grammar: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    **kwargs: Any,
) -> StructuredResult:
    """Smart auto-selection of the best structured output method.

    Priority:
    1. If grammar is provided → use grammar_output (most reliable)
    2. If tools are provided → use function_call_output
    3. If json_schema is provided → try grammar first, fall back to json_object
    4. Otherwise → use json_object_output

    Args:
        client: OpenAI client
        prompt: User prompt
        json_schema: Optional JSON Schema for structured output
        grammar: Optional GBNF grammar string (overrides schema)
        tools: Optional tool definitions for function calling
        model: Model name
        temperature: Sampling temperature
        max_tokens: Max output tokens
        **kwargs: Additional arguments

    Returns:
        StructuredResult from the best available method
    """
    logger.debug(
        f"🤖 [auto_structured] Selecting best format for prompt: {prompt[:80]}..."
    )

    # Priority 1: Explicit grammar
    if grammar:
        logger.debug("   → Using grammar_output (explicit grammar provided)")
        return grammar_output(
            client,
            prompt,
            grammar,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    # Priority 2: Function calling
    if tools:
        logger.debug("   → Using function_call_output (tools provided)")
        return function_call_output(
            client,
            prompt,
            tools,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    # Priority 3: JSON Schema → convert to grammar
    if json_schema:
        try:
            converted_grammar = grammar_from_json_schema(json_schema)
            logger.debug("   → Using grammar_output (converted from JSON Schema)")
            return grammar_output(
                client,
                prompt,
                converted_grammar,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                grammar_name="auto_from_schema",
                **kwargs,
            )
        except Exception as e:
            logger.warning(
                f"   → Grammar conversion failed: {e}, falling back to json_object"
            )

    # Priority 4: JSON object mode (fallback)
    logger.debug("   → Using json_object_output (default)")
    return json_object_output(
        client,
        prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )


# ─── Convenience Functions ─────────────────────────────────────────────────


def extract_person(client: OpenAI, text: str) -> StructuredResult:
    """Extract person info (name, age, city) from text using grammar."""
    return grammar_output(
        client,
        f"Extract the person's name, age, and city from: {text}",
        grammar=GRAMMAR_TEMPLATES["person"].grammar,
        grammar_name="person",
        temperature=0.0,
    )


def extract_list(client: OpenAI, text: str) -> StructuredResult:
    """Extract a list of items from text using grammar."""
    return grammar_output(
        client,
        f"Extract items as a JSON array from: {text}",
        grammar=GRAMMAR_TEMPLATES["list_of_strings"].grammar,
        grammar_name="list_of_strings",
        temperature=0.0,
    )
