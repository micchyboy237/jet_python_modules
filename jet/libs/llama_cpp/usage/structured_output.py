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
    BaseModel = object
    ValidationError = Exception

_VALIDATOR_BACKEND: str | None = None
_Draft202012Validator: Any = None

try:
    import jsonschema_rs

    _Draft202012Validator = jsonschema_rs.Draft202012Validator
    _VALIDATOR_BACKEND = "jsonschema-rs"
except ImportError:
    try:
        from jsonschema import Draft202012Validator

        _Draft202012Validator = Draft202012Validator
        _VALIDATOR_BACKEND = "jsonschema"
    except ImportError:
        _VALIDATOR_BACKEND = None

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
    validator_backend: str | None = None


_JSON_OBJECT_RE = re.compile(r"(\{.*\})", re.DOTALL)
_JSON_ARRAY_RE = re.compile(r"(\[.*\])", re.DOTALL)
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)


def extract_json(raw: str) -> dict | list | None:
    """Robustly extract JSON from model output, handling markdown fences."""
    stripped = raw.strip()

    # Try direct parse first
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # Try fenced code blocks
    match = _JSON_FENCE_RE.search(stripped)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try finding loose objects/arrays
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

    Returns:
        ResolvedFormat with api_format, schema, and optional system prompt.
    """
    if response_format is None:
        return ResolvedFormat(api_format=None, output_format=OutputFormat.TEXT)

    # Handle Pydantic Models
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

    # Handle Dict formats
    if isinstance(response_format, dict):
        fmt_type = response_format.get("type", "")

        # Grammar Mode
        if fmt_type == "grammar" or "grammar" in response_format:
            grammar_str = response_format.get("grammar", "")
            if not grammar_str:
                raise ValueError(
                    "Grammar response_format requires a 'grammar' key with GBNF string"
                )
            logger.debug("📜 Resolved grammar mode (will use extra_body.grammar)")
            return ResolvedFormat(
                api_format={"_grammar": grammar_str},
                output_format=OutputFormat.GRAMMAR,
            )

        # JSON Schema Object
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

        # JSON Schema Array
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

        # Passthrough formats
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
    """Generate a system prompt section describing expected JSON structure.

    Improvements:
    - Explicitly instructs to OMIT optional keys if no value exists (prevents nulls).
    - Handles 'anyOf' types common in Pydantic Optional fields.
    - Describes default values if present.
    """
    props = schema.get("properties", {})
    required = schema.get("required", [])

    lines = ["Return a JSON object with these exact fields:"]

    # Handle prefixItems for fixed-length tuples if present
    if "prefixItems" in schema.get("items", {}):
        prefix_items = schema["items"]["prefixItems"]
        lines.append("IMPORTANT: The 'items' field is a fixed-length tuple.")
        lines.append(
            f"Return an array with exactly {len(prefix_items)} elements in this order:"
        )
        for i, item_schema in enumerate(prefix_items):
            desc = item_schema.get("description", f"Item {i}")
            type_name = item_schema.get("type", "any")
            lines.append(f"  - Index {i}: {type_name} ({desc})")
        lines.append("Do NOT return objects inside this array. Return raw values.")
        lines.append("")

    for name, prop in props.items():
        is_required = name in required
        ptype = prop.get("type", "any")
        desc = prop.get("description", "")

        # Handle anyOf (common for Optional[T] in Pydantic)
        if "anyOf" in prop:
            types_list = []
            for item in prop["anyOf"]:
                t = item.get("type", "any")
                if t == "null":
                    continue  # We handle nullability via "optional" label
                types_list.append(t)
            ptype = " | ".join(types_list) if types_list else "any"

        # Build the field description line
        if is_required:
            req_mark = " (REQUIRED)"
        else:
            # Check for defaults to give better hints
            default_val = prop.get("default")
            if default_val is not None:
                req_mark = f" (optional, default: {json.dumps(default_val)})"
            else:
                req_mark = " (optional, OMIT key if no value)"

        line_prefix = f'  - "{name}": {ptype}{req_mark}'

        if desc:
            lines.append(f"{line_prefix}\n    {desc}")
        else:
            lines.append(line_prefix)

    if required:
        lines.append(f"\nRequired fields: {', '.join(required)}")

    lines.append("\nReturn ONLY valid JSON, no markdown, no explanation.")
    return "\n".join(lines)


def parse_structured_content(
    content: str,
    resolved: ResolvedFormat,
) -> StructuredResult:
    """Parse and validate raw model output against a resolved format.

    Validation priority:
      1. Pydantic model_validate (if model_type set)
      2. Modern JSON Schema validation (jsonschema-rs or jsonschema fallback)
      3. Grammar mode (trusted via GBNF constraints)
      4. Fallback: raw JSON extraction only
    """
    if resolved.output_format == OutputFormat.TEXT:
        return StructuredResult(
            success=True,
            content=content,
            format_used=OutputFormat.TEXT,
            validator_backend=None,
        )

    extracted = extract_json(content)
    if extracted is None:
        logger.warning("⚠️ Failed to extract JSON from response")
        return StructuredResult(
            success=False,
            content=content,
            error="Failed to extract JSON from response",
            format_used=resolved.output_format,
            validator_backend=_VALIDATOR_BACKEND,
        )

    # 1. Pydantic Validation
    if resolved.model_type is not None and PYDANTIC_AVAILABLE:
        try:
            instance = resolved.model_type.model_validate(extracted)
            logger.debug(
                f"✅ Validated against {resolved.model_type.__name__} (pydantic)"
            )
            return StructuredResult(
                success=True,
                content=content,
                parsed=instance,
                format_used=resolved.output_format,
                validator_backend="pydantic",
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
                validator_backend="pydantic",
            )

    # 2. JSON Schema Validation
    if resolved.schema is not None and _Draft202012Validator is not None:
        try:
            validator = _Draft202012Validator(resolved.schema)
            errors_list = list(validator.iter_errors(extracted))
            if errors_list:
                validation_msgs = [
                    f"{'.'.join(str(p) for p in err.absolute_path) or '(root)'}: {err.message}"
                    for err in errors_list
                ]
                logger.warning(
                    f"⚠️ JSON Schema validation failed ({_VALIDATOR_BACKEND}): "
                    f"{validation_msgs}"
                )
                return StructuredResult(
                    success=False,
                    content=content,
                    parsed=extracted,
                    error=f"JSON Schema validation failed ({len(errors_list)} errors)",
                    format_used=resolved.output_format,
                    validation_errors=validation_msgs,
                    validator_backend=_VALIDATOR_BACKEND,
                )
            logger.info(
                f"✅ Structured output validated via {_VALIDATOR_BACKEND} (Draft 2020-12)"
            )
            return StructuredResult(
                success=True,
                content=content,
                parsed=extracted,
                format_used=resolved.output_format,
                validator_backend=_VALIDATOR_BACKEND,
            )
        except Exception as exc:
            logger.error(
                f"❌ JSON Schema validator ({_VALIDATOR_BACKEND}) raised: {exc}"
            )
            return StructuredResult(
                success=False,
                content=content,
                parsed=extracted,
                error=f"Validator error ({_VALIDATOR_BACKEND}): {exc}",
                format_used=resolved.output_format,
                validator_backend=_VALIDATOR_BACKEND,
            )

    # 3. Grammar Mode (Trusted)
    if resolved.output_format == OutputFormat.GRAMMAR:
        logger.debug("✅ Grammar-constrained output accepted (GBNF validated)")
        return StructuredResult(
            success=True,
            content=content,
            parsed=extracted,
            format_used=OutputFormat.GRAMMAR,
            validator_backend="gbnf",
        )

    # 4. Fallback
    logger.debug(
        "⚠️ No JSON Schema validator installed; returning extracted JSON without validation"
    )
    return StructuredResult(
        success=True,
        content=content,
        parsed=extracted,
        format_used=resolved.output_format,
        validator_backend=None,
    )
