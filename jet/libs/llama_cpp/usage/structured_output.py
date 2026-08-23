# jet_python_modules/jet/libs/llama_cpp/usage/structured_output.py
"""Pure structured output validation and schema resolution utilities.

This module contains NO streaming or API call logic. It provides:
  - resolve_response_format(): Normalize Pydantic/Schema/Dict → API-ready format
  - parse_structured_content(): Validate raw text against a target format
  - build_schema_prompt(): Generate system prompts for schema adherence

All streaming/orchestration happens in chat_stream_observability.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, Type, TypeVar

try:
    from pydantic import BaseModel, ValidationError

    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object  # type: ignore[misc,assignment]
    ValidationError = Exception  # type: ignore[misc,assignment]

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class OutputFormat(Enum):
    """Supported structured output modes."""

    JSON_OBJECT = "json_object"
    JSON_SCHEMA = "json_schema"
    GRAMMAR = "grammar"
    TEXT = "text"


@dataclass
class ResolvedFormat:
    """Normalized response format ready for the OpenAI API."""

    api_format: dict[str, Any] | None
    output_format: OutputFormat
    schema: dict[str, Any] | None = None
    model_type: Type[BaseModel] | None = None
    system_prompt_addition: str | None = None


@dataclass
class StructuredResult(Generic[T]):
    """Unified result from structured output parsing."""

    success: bool
    content: str
    parsed: dict | list | T | None = None
    error: str | None = None
    format_used: OutputFormat = OutputFormat.TEXT
    validation_errors: list[str] = field(default_factory=list)


_JSON_OBJECT_RE = re.compile(r"(\{.*\})", re.DOTALL)
_JSON_ARRAY_RE = re.compile(r"(\[.*\])", re.DOTALL)
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def extract_json(raw: str) -> dict | list | None:
    """Robustly extract JSON from model output, handling markdown fences."""
    stripped = raw.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    match = _JSON_FENCE_RE.search(stripped)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

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


def resolve_response_format(
    response_format: Any,
) -> ResolvedFormat:
    """Normalize user-provided response format into API-ready structure.

    Accepts:
      - None → text mode
      - Pydantic BaseModel subclass → auto-generate json_schema + prompt
      - dict with 'properties' or '$schema' → JSON Schema (object)
      - dict with 'type': 'array' and 'items' → JSON Schema (array)
      - dict with 'type': 'json_object' → passthrough
      - dict with 'type': 'json_schema' → passthrough
      - dict with 'type': 'grammar' or 'grammar' key → grammar via extra_body
        (NOT sent as response_format; caller must merge into extra_body_params)

    Returns:
        ResolvedFormat with api_format, schema, and optional system prompt.
        For grammar mode, api_format is None and the grammar string is stored
        in resolved_fmt.api_format under a special key for the caller to handle.
    """
    if response_format is None:
        return ResolvedFormat(api_format=None, output_format=OutputFormat.TEXT)

    # ── Pydantic model class ──────────────────────────────────────────
    if (
        PYDANTIC_AVAILABLE
        and isinstance(response_format, type)
        and issubclass(response_format, BaseModel)
    ):
        schema = response_format.model_json_schema()
        api_format = {
            "type": "json_schema",
            "json_schema": {
                "name": response_format.__name__,
                "strict": True,
                "schema": schema,
            },
        }
        prompt_addition = build_schema_prompt(schema)
        logger.debug(
            f"📐 Resolved Pydantic model '{response_format.__name__}' → json_schema"
        )
        return ResolvedFormat(
            api_format=api_format,
            output_format=OutputFormat.JSON_SCHEMA,
            schema=schema,
            model_type=response_format,
            system_prompt_addition=prompt_addition,
        )

    # ── Dict-based formats ────────────────────────────────────────────
    if isinstance(response_format, dict):
        fmt_type = response_format.get("type", "")

        # Grammar mode: extract grammar string, do NOT set api_format
        # Grammar goes in extra_body, not response_format
        if fmt_type == "grammar" or "grammar" in response_format:
            grammar_str = response_format.get("grammar", "")
            if not grammar_str:
                raise ValueError(
                    "Grammar response_format requires a 'grammar' key with GBNF string"
                )
            logger.debug("📜 Resolved grammar mode (will use extra_body.grammar)")
            return ResolvedFormat(
                api_format={"_grammar": grammar_str},  # sentinel for caller
                output_format=OutputFormat.GRAMMAR,
            )

        # JSON Schema dict (object): has 'properties' or '$schema'
        if "properties" in response_format or "$schema" in response_format:
            name = response_format.get("title", "custom_schema")
            api_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": name,
                    "strict": True,
                    "schema": response_format,
                },
            }
            prompt_addition = build_schema_prompt(response_format)
            logger.debug(f"📐 Resolved JSON Schema dict → json_schema ({name})")
            return ResolvedFormat(
                api_format=api_format,
                output_format=OutputFormat.JSON_SCHEMA,
                schema=response_format,
                system_prompt_addition=prompt_addition,
            )

        # JSON Schema dict (array): has 'type': 'array' and 'items'
        if fmt_type == "array" and "items" in response_format:
            name = response_format.get("title", "array_schema")
            api_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": name,
                    "strict": True,
                    "schema": response_format,
                },
            }
            # Build prompt for array items
            items_schema = response_format["items"]
            if isinstance(items_schema, dict) and "properties" in items_schema:
                prompt_addition = build_schema_prompt(items_schema)
                prompt_addition += (
                    "\n\nIMPORTANT: Return a JSON ARRAY of objects matching "
                    "the schema above, e.g. [{...}, {...}]."
                )
            else:
                prompt_addition = "Return a JSON ARRAY. Each element should match the expected schema."
            logger.debug(f"📐 Resolved JSON Schema array → json_schema ({name})")
            return ResolvedFormat(
                api_format=api_format,
                output_format=OutputFormat.JSON_SCHEMA,
                schema=response_format,
                system_prompt_addition=prompt_addition,
            )

        # Known passthrough types
        if fmt_type in ("json_object", "json_schema"):
            logger.debug(f"📐 Resolved dict format: {fmt_type}")
            return ResolvedFormat(
                api_format=response_format,
                output_format=OutputFormat(fmt_type),
            )

    raise ValueError(
        f"Unsupported response_format: {type(response_format).__name__}. "
        f"Expected None, dict, Pydantic model, or JSON Schema dict."
    )


def build_schema_prompt(schema: dict[str, Any]) -> str:
    """Generate a system prompt section describing expected JSON structure."""
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
    if required:
        lines.append(f"\nRequired fields: {', '.join(required)}")
    lines.append("Return ONLY valid JSON, no markdown, no explanation.")
    return "\n".join(lines)


def parse_structured_content(
    content: str,
    resolved: ResolvedFormat,
) -> StructuredResult:
    """Parse and validate raw model output against a resolved format.

    This is a pure function — no API calls, no streaming.
    Testable independently with any string input.
    """
    if resolved.output_format == OutputFormat.TEXT:
        return StructuredResult(
            success=True, content=content, format_used=OutputFormat.TEXT
        )

    extracted = extract_json(content)
    if extracted is None:
        return StructuredResult(
            success=False,
            content=content,
            error="Failed to extract JSON from response",
            format_used=resolved.output_format,
        )

    # Pydantic validation
    if resolved.model_type is not None and PYDANTIC_AVAILABLE:
        try:
            instance = resolved.model_type.model_validate(extracted)
            logger.debug(f"✅ Validated against {resolved.model_type.__name__}")
            return StructuredResult(
                success=True,
                content=content,
                parsed=instance,
                format_used=resolved.output_format,
            )
        except ValidationError as e:
            errors = [f"{err['loc']}: {err['msg']}" for err in e.errors()]
            logger.warning(f"⚠️ Pydantic validation failed: {errors}")
            return StructuredResult(
                success=False,
                content=content,
                parsed=extracted,
                error="Pydantic validation failed",
                format_used=resolved.output_format,
                validation_errors=errors,
            )

    # Plain JSON success
    return StructuredResult(
        success=True,
        content=content,
        parsed=extracted,
        format_used=resolved.output_format,
    )
