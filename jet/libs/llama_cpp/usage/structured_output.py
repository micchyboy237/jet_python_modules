# jet_python_modules/jet/libs/llama_cpp/usage/structured_output.py
"""Structured output helpers for llama.cpp OpenAI-compatible server.

Provides encapsulated, reusable functions for response formats that
actually work with llama.cpp:

  - json_object_output()     → Best-effort JSON via response_format
  - auto_structured()        → Smart auto-selection based on requirements

Pydantic Integration:
  - pydantic_output()        → Extract Pydantic model via json_object mode
  - pydantic_list_output()   → Extract list of Pydantic models
  - parsed_completion()      → OpenAI-compatible interface (result.parsed)
  - pydantic_to_json_schema() → Convert Pydantic model → JSON Schema

Design principles:
  - Each function returns a typed dataclass, not raw dict/str
  - Built-in JSON extraction handles common llama.cpp quirks (markdown fences, etc.)
  - Comprehensive logging at every step
  - Pydantic models work via json_object mode with schema-enhanced prompting
  - OpenAI's pydantic_function_tool() pattern is replicated via parsed_completion()
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Generic, Type, TypeVar

from jet.libs.llama_cpp.usage.chat_stream_observability import (
    MODEL as DEFAULT_MODEL,
)
from jet.libs.llama_cpp.usage.chat_stream_observability import (
    run_chat_stream,
)
from openai import OpenAI

logger = logging.getLogger(Path(__file__).stem)

# ─── Pydantic Availability Check ──────────────────────────────────────────

try:
    from pydantic import BaseModel

    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object  # type: ignore

T = TypeVar("T", bound=BaseModel)


# ─── Data Classes ──────────────────────────────────────────────────────────


class OutputFormat(Enum):
    """Supported output formats for llama.cpp."""

    JSON_OBJECT = "json_object"


@dataclass
class StructuredResult:
    """Unified result from any structured output method.

    Attributes:
        format_used: Which format produced this result
        success: Whether valid structured output was obtained
        content: Raw text content from the model
        parsed: Parsed JSON (if applicable)
        usage: Token usage stats
        finish_reason: Why model stopped generating
        duration_ms: Total round-trip time in milliseconds
        error: Error message if failed
    """

    format_used: OutputFormat
    success: bool
    content: str
    parsed: dict | list | None = None
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
            "usage": self.usage,
            "finish_reason": self.finish_reason,
            "duration_ms": self.duration_ms,
            "error": self.error,
        }


@dataclass
class PydanticResult(Generic[T]):
    """Structured result containing a parsed Pydantic model instance.

    Type Parameters:
        T: The Pydantic model type

    Attributes:
        success: Whether a valid model instance was produced
        model: The parsed Pydantic model instance (None if failed)
        raw_result: The underlying StructuredResult for debugging
        validation_errors: Pydantic validation errors if parsing failed
    """

    success: bool
    model: T | list[T] | None = None
    raw_result: StructuredResult | None = None
    validation_errors: list[str] = field(default_factory=list)

    @property
    def content(self) -> str:
        """Raw text from the model."""
        return self.raw_result.content if self.raw_result else ""

    @property
    def usage(self) -> dict[str, int] | None:
        """Token usage if available."""
        return self.raw_result.usage if self.raw_result else None


class ParsedOutput(Generic[T]):
    """Mimics OpenAI's ParsedChatCompletion pattern.

    Provides a familiar interface if you're used to the OpenAI SDK's
    parse parameter with pydantic_function_tool().

    Usage:
        result = parsed_completion(client, prompt, MyModel)
        print(result.parsed.name)  # Direct model access
    """

    content: str
    parsed: T | None
    usage: dict[str, int] | None
    finish_reason: str | None

    def __init__(
        self,
        content: str,
        parsed: T | None,
        usage: dict[str, int] | None = None,
        finish_reason: str | None = None,
    ):
        self.content = content
        self.parsed = parsed
        self.usage = usage
        self.finish_reason = finish_reason

    def __repr__(self) -> str:
        return f"ParsedOutput(parsed={type(self.parsed).__name__ if self.parsed else None})"


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

    # Attempt 3: Find last complete JSON object/array
    for pattern in [_JSON_OBJECT_RE, _JSON_ARRAY_RE]:
        matches = pattern.findall(stripped)
        for candidate in reversed(matches):
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, (dict, list)):
                    return parsed
            except json.JSONDecodeError:
                continue

    return None


# ─── JSON Schema Utilities ─────────────────────────────────────────────────


def pydantic_to_json_schema(model: Type[BaseModel]) -> dict[str, Any]:
    """Convert a Pydantic model to JSON Schema.

    Args:
        model: Pydantic BaseModel subclass

    Returns:
        JSON Schema dict

    Raises:
        ImportError: If pydantic is not installed
    """
    if not PYDANTIC_AVAILABLE:
        raise ImportError("pydantic is required. Install with: pip install pydantic")

    schema = model.model_json_schema()
    logger.debug(
        f"📋 Generated JSON Schema for {model.__name__}: "
        f"{json.dumps(schema, indent=2)[:200]}..."
    )
    return schema


def build_schema_prompt(schema: dict[str, Any]) -> str:
    """Build an enhanced prompt section describing the expected JSON structure.

    Args:
        schema: JSON Schema dict

    Returns:
        Prompt string describing the expected fields
    """
    props = schema.get("properties", {})
    required = schema.get("required", [])

    lines = ["Return a JSON object with these exact fields:"]
    for name, prop in props.items():
        ptype = prop.get("type", "string")
        desc = prop.get("description", "")
        req_mark = " (required)" if name in required else " (optional)"
        lines.append(f'  - "{name}": {ptype}{req_mark}')
        if desc:
            lines.append(f"    {desc}")

    lines.append("\nRequired fields: " + ", ".join(required))
    lines.append("Return ONLY the JSON object, no markdown, no explanation.")
    return "\n".join(lines)


# ─── Core Output Function ─────────────────────────────────────────────────


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


# ─── Pydantic-Aware Output Functions ──────────────────────────────────────


def pydantic_output(
    client: OpenAI,
    prompt: str,
    model_type: Type[T],
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    **kwargs: Any,
) -> PydanticResult[T]:
    """Get output validated against a Pydantic model.

    Uses json_object mode with schema-enhanced prompting for reliable
    structured output on llama.cpp.

    Args:
        client: OpenAI client pointing to llama.cpp server
        prompt: User prompt describing what to extract/generate
        model_type: Pydantic BaseModel subclass defining the expected structure
        model: llama.cpp model name
        temperature: Low temperature for deterministic output (0.0-0.3)
        max_tokens: Maximum output tokens
        **kwargs: Additional arguments passed to the underlying method

    Returns:
        PydanticResult with the validated model instance

    Raises:
        ImportError: If pydantic is not installed

    Example:
        >>> class Person(BaseModel):
        ...     name: str
        ...     age: int
        ...     city: str
        ...
        >>> result = pydantic_output(client, "Extract: John, 42, SF", Person)
        >>> if result.success:
        ...     print(result.model.name)  # "John"
        ...     print(result.model.age)   # 42
    """
    if not PYDANTIC_AVAILABLE:
        raise ImportError("pydantic is required. Install with: pip install pydantic")

    schema = pydantic_to_json_schema(model_type)
    model_fields = ", ".join(
        f'"{name}" ({prop.get("type", "any")})'
        for name, prop in schema.get("properties", {}).items()
    )

    logger.debug(
        f"🏗️ [pydantic_output] Model={model_type.__name__}, fields=({model_fields})"
    )

    # Build enhanced prompt with schema information
    schema_prompt = build_schema_prompt(schema)
    enhanced_prompt = f"{prompt}\n\n{schema_prompt}"

    raw = json_object_output(
        client,
        enhanced_prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )

    # Validate against Pydantic model
    if raw.success and raw.parsed:
        try:
            instance = model_type.model_validate(raw.parsed)
            logger.debug(f"✅ [pydantic_output] Validated as {model_type.__name__}")
            return PydanticResult(
                success=True,
                model=instance,
                raw_result=raw,
            )
        except Exception as e:
            logger.warning(
                f"⚠️ [pydantic_output] Validation failed: {e}\n"
                f"   Parsed: {json.dumps(raw.parsed)[:200]}"
            )
            return PydanticResult(
                success=False,
                raw_result=raw,
                validation_errors=[str(e)],
            )
    else:
        return PydanticResult(
            success=False,
            raw_result=raw,
            validation_errors=["No valid JSON parsed from response"],
        )


def pydantic_list_output(
    client: OpenAI,
    prompt: str,
    item_type: Type[T],
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    **kwargs: Any,
) -> PydanticResult[list[T]]:
    """Get a list of Pydantic model instances from the output.

    Useful for extracting multiple entities from text.

    Args:
        client: OpenAI client
        prompt: User prompt asking for a list
        item_type: Pydantic model for each list item
        model: Model name
        temperature: Sampling temperature
        max_tokens: Max output tokens
        **kwargs: Additional arguments

    Returns:
        PydanticResult containing list of validated model instances

    Raises:
        ImportError: If pydantic is not installed

    Example:
        >>> class Person(BaseModel):
        ...     name: str
        ...     age: int
        ...
        >>> result = pydantic_list_output(
        ...     client, "List all people mentioned", Person
        ... )
        >>> for person in result.model:
        ...     print(person.name)
    """
    if not PYDANTIC_AVAILABLE:
        raise ImportError("pydantic is required")

    schema = pydantic_to_json_schema(item_type)

    # Build enhanced prompt requesting a JSON array of objects
    schema_prompt = build_schema_prompt(schema)
    enhanced_prompt = (
        f"{prompt}\n\n"
        f"{schema_prompt}\n\n"
        f"IMPORTANT: Return a JSON ARRAY of objects, e.g. [{{...}}, {{...}}]. "
        f"Each object in the array should match the format above."
    )

    logger.debug(f"📋 [pydantic_list_output] List of {item_type.__name__}")

    raw = json_object_output(
        client,
        enhanced_prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )

    if raw.success and raw.parsed:
        if isinstance(raw.parsed, dict):
            items_to_validate = [raw.parsed]
        elif isinstance(raw.parsed, list):
            items_to_validate = raw.parsed
        else:
            items_to_validate = []

        validated: list[T] = []
        errors: list[str] = []
        for i, item in enumerate(items_to_validate):
            try:
                validated.append(item_type.model_validate(item))
            except Exception as e:
                errors.append(f"Item {i}: {e}")

        success = len(validated) > 0
        logger.debug(
            f"{'✅' if success else '⚠️'} [pydantic_list_output] "
            f"Validated {len(validated)}/{len(items_to_validate)} items"
        )

        return PydanticResult(
            success=success,
            model=validated if success else None,
            raw_result=raw,
            validation_errors=errors,
        )
    else:
        return PydanticResult(
            success=False,
            raw_result=raw,
            validation_errors=["No valid JSON array parsed"],
        )


def parsed_completion(
    client: OpenAI,
    prompt: str,
    response_model: Type[T],
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    **kwargs: Any,
) -> ParsedOutput[T]:
    """OpenAI-compatible parsed completion for llama.cpp.

    Mimics the pattern:
        client.chat.completions.create(
            ...,
            response_format=pydantic_function_tool(MyModel),
        )
    →  result.choices[0].message.parsed  # MyModel instance

    But works with llama.cpp using json_object mode internally.

    Args:
        client: OpenAI client
        prompt: User prompt
        response_model: Pydantic model class
        model: Model name
        temperature: Sampling temperature
        max_tokens: Max output tokens
        **kwargs: Additional arguments

    Returns:
        ParsedOutput with .parsed attribute containing model instance

    Raises:
        ImportError: If pydantic is not installed

    Example:
        >>> class Person(BaseModel):
        ...     name: str
        ...     age: int
        ...
        >>> result = parsed_completion(client, "Extract person info: ...", Person)
        >>> if result.parsed:
        ...     print(result.parsed.name)
    """
    if not PYDANTIC_AVAILABLE:
        raise ImportError("pydantic is required. Install with: pip install pydantic")

    pydantic_result = pydantic_output(
        client,
        prompt,
        response_model,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )

    return ParsedOutput(
        content=pydantic_result.content,
        parsed=pydantic_result.model,  # type: ignore
        usage=pydantic_result.usage,
        finish_reason=pydantic_result.raw_result.finish_reason
        if pydantic_result.raw_result
        else None,
    )
