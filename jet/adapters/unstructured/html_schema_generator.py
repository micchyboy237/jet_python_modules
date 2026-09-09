"""Reusable HTML-to-Schema generator for LLM RAG traversal using Unstructured."""

import logging
from typing import Any

from unstructured.partition.html import partition_html

logger = logging.getLogger(__name__)


# Elements typically not useful for RAG content retrieval
DEFAULT_EXCLUDE_TYPES = {"PageBreak", "Header", "Footer", "PageNumber"}


def generate_html_schema(
    html_content: str,
    exclude_types: set[str] | None = None,
    include_metadata: bool = True,
) -> list[dict[str, Any]]:
    """
    Parse HTML and generate a hierarchical schema for LLM RAG traversal.

    Args:
        html_content: Raw HTML string to parse.
        exclude_types: Element types to exclude from schema. Defaults to DEFAULT_EXCLUDE_TYPES.
        include_metadata: Whether to include element metadata in schema nodes.

    Returns:
        List of root-level schema nodes with nested children.
    """
    if exclude_types is None:
        exclude_types = DEFAULT_EXCLUDE_TYPES

    logger.info("Partitioning HTML content...")
    elements = partition_html(text=html_content)
    logger.info("Partitioned %d elements", len(elements))

    # Build lookup maps
    element_map: dict[str, dict[str, Any]] = {}
    roots: list[dict[str, Any]] = []

    # First pass: create node skeletons
    for el in elements:
        el_dict = el.to_dict()
        el_type = el_dict.get("type", "")

        if el_type in exclude_types:
            logger.debug("Excluding element type: %s", el_type)
            continue

        node: dict[str, Any] = {
            "element_id": el_dict["element_id"],
            "type": el_type,
            "text": el_dict.get("text", ""),
            "children": [],
        }

        if include_metadata:
            node["metadata"] = {
                k: v
                for k, v in el_dict.get("metadata", {}).items()
                if k
                in (
                    "category_depth",
                    "parent_id",
                    "link_urls",
                    "link_texts",
                    "languages",
                )
            }

        element_map[node["element_id"]] = node

    # Second pass: build hierarchy using parent_id
    for node in element_map.values():
        parent_id = node.get("metadata", {}).get("parent_id")
        if parent_id and parent_id in element_map:
            element_map[parent_id]["children"].append(node)
            logger.debug("Linked %s as child of %s", node["element_id"], parent_id)
        else:
            roots.append(node)

    logger.info("Generated schema with %d root nodes", len(roots))
    return roots


def schema_to_llm_context(schema: list[dict[str, Any]], max_depth: int = 10) -> str:
    """
    Convert schema tree to indented text representation for LLM context injection.

    Args:
        schema: Hierarchical schema from generate_html_schema.
        max_depth: Maximum traversal depth to prevent runaway recursion.

    Returns:
        Indented string representation of the document structure.
    """
    lines: list[str] = []

    def _walk(nodes: list[dict[str, Any]], depth: int) -> None:
        if depth > max_depth:
            return
        indent = "  " * depth
        for node in nodes:
            el_type = node.get("type", "Unknown")
            text_preview = (node.get("text", "") or "")[:200]
            lines.append(f"{indent}[{el_type}] {text_preview}")
            _walk(node.get("children", []), depth + 1)

    _walk(schema, 0)
    return "\n".join(lines)
