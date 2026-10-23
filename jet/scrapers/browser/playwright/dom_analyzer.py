"""
DOM Analyzer for Agentic Flows
Provides high-level structural analysis of web pages to help LLMs understand
patterns (lists/grids) and density (main content areas).
"""

import asyncio
from typing import List, Literal, Optional, TypedDict

from jet.logger import logger
from jet.scrapers.utils.base import (
    ParentWithCommonClass,
    SimpleParent,
    get_flattened_parents_with_most_children,
    get_parents_with_common_class,
)


class PatternResult(TypedDict):
    """Result from the Pattern Recognition Engine."""

    common_class: str
    common_tag: str
    children_count: int
    parent_tag: str
    parent_id: str
    sample_text: str


class StructuralMapResult(TypedDict):
    """Result from the Structural Map & Density Analyzer."""

    node_id: str
    tag: str
    depth: int
    child_count: int
    child_xpaths: List[str]
    child_css_selectors: List[str]


async def analyze_patterns(
    source: str,
    class_name: Optional[str] = None,
    match_type: Literal["tag", "class", "both"] = "both",
    headless: bool = True,
    timeout_ms: int = 10000,
) -> List[PatternResult]:
    """
    The 'Pattern Recognition' Engine.
    Identifies repeating structures (lists, grids, cards) in the DOM.

    Args:
        source: URL, file path, or raw HTML string.
        class_name: Optional specific class to look for.
        match_type: How to match children ('tag', 'class', or 'both').
        headless: Run browser in headless mode.
        timeout_ms: Page load timeout.

    Returns:
        List of PatternResult dicts sorted by significance (children count).
    """
    logger.info(f"Analyzing patterns for source: {source[:50]}...")

    # Run synchronous analysis in a thread since it uses sync Playwright internally
    parents: List[ParentWithCommonClass] = await asyncio.to_thread(
        get_parents_with_common_class,
        source=source,
        class_name=class_name,
        match_type=match_type,
        headless=headless,
        timeout_ms=timeout_ms,
    )

    results: List[PatternResult] = []
    for p in parents:
        sample_text = ""
        if p.children:
            first_child = p.children[0]
            sample_text = (first_child.text or "")[:100]

        results.append(
            {
                "common_class": p.common_class,
                "common_tag": p.common_tag,
                "children_count": p.children_count,
                "parent_tag": p.tag,
                "parent_id": p.id,
                "sample_text": sample_text,
            }
        )

    logger.info(f"Found {len(results)} significant patterns.")
    return results


async def analyze_structure(
    source: str,
    min_children: int = 5,
    headless: bool = True,
    timeout_ms: int = 10000,
) -> List[StructuralMapResult]:
    """
    The 'Structural Map' & 'Density Analyzer'.
    Identifies dense areas of the DOM to help the LLM focus on main content.

    Args:
        source: URL, file path, or raw HTML string.
        min_children: Minimum number of direct children to consider a node 'dense'.
        headless: Run browser in headless mode.
        timeout_ms: Page load timeout.

    Returns:
        List of StructuralMapResult dicts sorted by density (child count).
    """
    logger.info(f"Analyzing structure/density for source: {source[:50]}...")

    # Run synchronous analysis in a thread since it uses sync Playwright internally
    parents: List[SimpleParent] = await asyncio.to_thread(
        get_flattened_parents_with_most_children,
        source=source,
        min_children=min_children,
        headless=headless,
        timeout_ms=timeout_ms,
    )

    results: List[StructuralMapResult] = []
    for p in parents:
        results.append(
            {
                "node_id": p["node_id"],
                "tag": p["tag"],
                "depth": p["depth"],
                "child_count": p["child_count"],
                "child_xpaths": p["child_xpaths"][:5],
                "child_css_selectors": p["child_css_selectors"][:5],
            }
        )

    logger.info(f"Found {len(results)} dense structural nodes.")
    return results
