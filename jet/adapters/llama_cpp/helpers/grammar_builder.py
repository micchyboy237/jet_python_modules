# jet_python_modules/jet/adapters/llama_cpp/helpers/grammar_builder.py
"""Reusable grammar builder using llama.cpp's json_schema_to_grammar.

Wraps SchemaConverter to generate GBNF grammars from JSON Schema dicts
at runtime. Designed for dynamic schemas where the shape depends on
input data (e.g., variable-length arrays with per-index const values).
"""

from __future__ import annotations

from typing import Any

from jet.adapters.llama_cpp.helpers.json_schema_to_grammar import SchemaConverter
from jet.logger import logger


def build_grammar_from_schema(
    schema: dict[str, Any],
    *,
    prop_order: dict[str, int] | None = None,
) -> str:
    """Generate a GBNF grammar string from a JSON Schema dict.

    Args:
        schema: Valid JSON Schema dict. Supports prefixItems, const, enum,
            object properties, and all features in llama.cpp's converter.
        prop_order: Optional property name → position mapping for object
            field ordering in generated grammar.

    Returns:
        Valid GBNF grammar string ready for llama-server's grammar parameter.

    Raises:
        ValueError: If schema conversion fails.
    """
    try:
        converter = SchemaConverter(
            prop_order=prop_order or {},
            allow_fetch=False,
            dotall=False,
            raw_pattern=False,
        )
        converter.visit(schema, "")
        grammar = converter.format_grammar()
        logger.debug(f"Generated GBNF grammar ({len(grammar)} bytes)")
        return grammar
    except Exception as e:
        raise ValueError(f"Failed to generate grammar from schema: {e}") from e


def validate_grammar(grammar: str) -> str | None:
    """Validate GBNF grammar client-side via gbnf.dev if available.

    Returns None if valid or if gbnf is not installed.
    Returns error message string if validation fails.
    """
    try:
        from gbnf import validate_grammar as _validate  # type: ignore[import-untyped]

        result = _validate(grammar)
        if result is True or result is None:
            return None
        return str(result)
    except ImportError:
        logger.debug("gbnf package not installed; skipping grammar validation")
        return None
    except Exception as e:
        logger.warning(f"Grammar validation check failed (non-fatal): {e}")
        return None
