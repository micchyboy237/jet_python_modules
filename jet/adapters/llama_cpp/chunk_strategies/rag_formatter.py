"""RAG formatting utilities for chunk post-processing.

Design principle: Prefer explicit content_type metadata over heuristic detection.
Fallback heuristics are provided for backward compatibility but logged as warnings.
"""

import logging
import re
from typing import List, Literal, Optional

logger = logging.getLogger(__name__)

ContentType = Literal["code", "table", "prose", "unknown"]

# Markers use double brackets to reduce collision with natural text
_CODE_OPEN = "[[CODE]]"
_CODE_CLOSE = "[[/CODE]]"
_TABLE_OPEN = "[[TABLE]]"
_TABLE_CLOSE = "[[/TABLE]]"

# Heuristic patterns (fallback only)
_CODE_PATTERN = re.compile(r"(?:^|\n)\s*(?:```|def\s+\w+|class\s+\w+|import\s+\w+)")
_TABLE_PATTERN = re.compile(r"(?:^|\n)\s*\|.+\|")


def detect_content_type(text: str) -> ContentType:
    """Heuristic content type detection. Use only when metadata is unavailable."""
    if _CODE_PATTERN.search(text):
        return "code"
    if _TABLE_PATTERN.search(text):
        return "table"
    return "prose"


def format_chunks_for_rag(
    chunks: List[str],
    content_types: Optional[List[ContentType]] = None,
) -> List[str]:
    """Format chunks with RAG-specific structural markers.

    Args:
        chunks: List of chunk strings.
        content_types: Optional parallel list of content types. If None,
            falls back to heuristic detection (logged as warning).

    Returns:
        Formatted chunks with structural markers.
    """
    if content_types is None:
        logger.warning(
            "format_chunks_for_rag called without content_types; "
            "using heuristic detection for %d chunks (may be inaccurate)",
            len(chunks),
        )
        content_types = [detect_content_type(c) for c in chunks]

    if len(content_types) != len(chunks):
        raise ValueError(
            f"content_types length ({len(content_types)}) != chunks length ({len(chunks)})"
        )

    formatted: List[str] = []
    for i, (chunk, ctype) in enumerate(zip(chunks, content_types)):
        if ctype == "code":
            formatted.append(f"{_CODE_OPEN}\n{chunk}\n{_CODE_CLOSE}")
            logger.debug("Chunk %d: wrapped as CODE", i)
        elif ctype == "table":
            formatted.append(f"{_TABLE_OPEN}\n{chunk}\n{_TABLE_CLOSE}")
            logger.debug("Chunk %d: wrapped as TABLE", i)
        else:
            formatted.append(chunk)
            logger.debug("Chunk %d: kept as PROSE", i)

    logger.info(
        "Formatted %d chunks: %d code, %d table, %d prose",
        len(chunks),
        content_types.count("code"),
        content_types.count("table"),
        content_types.count("prose") + content_types.count("unknown"),
    )
    return formatted
