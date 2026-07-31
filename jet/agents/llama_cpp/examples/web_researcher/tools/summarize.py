# web_researcher/tools/summarize.py
"""
Text summarization tool using the LLM directly.
"""

import logging
from typing import Optional

from smolagents import tool

logger = logging.getLogger(__name__)


@tool
def summarize_text(
    text: str,
    max_length: Optional[int] = 300,
    focus: Optional[str] = None,
) -> str:
    """
    Summarizes text to extract key information.

    Args:
        text: The text to summarize.
        max_length: Maximum words in summary (default: 300).
        focus: Specific aspect to focus on (e.g., "facts", "numbers", "causes").

    Returns:
        Summarized text.
    """
    logger.info(f"Summarizing text ({len(text)} characters)")

    # Simple extractive summarization - take first and last paragraphs
    paragraphs = text.split("\n\n")
    if len(paragraphs) <= 2:
        return text[: max_length * 4]

    # Get first paragraph (intro) and last paragraph (conclusion)
    summary_parts = [paragraphs[0], paragraphs[-1]]

    # If focus is specified, try to find relevant paragraphs
    if focus and len(paragraphs) > 3:
        focus_lower = focus.lower()
        focus_terms = focus_lower.split()

        for i, p in enumerate(paragraphs[1:-1], 1):
            p_lower = p.lower()
            if any(term in p_lower for term in focus_terms):
                summary_parts.insert(1, p)
                if len(summary_parts) >= 3:
                    break

    result = "\n\n".join(summary_parts)

    # Truncate to max length
    words = result.split()
    if len(words) > max_length:
        result = " ".join(words[:max_length]) + "..."

    logger.info(f"Summary generated: {len(result)} characters")
    return result
