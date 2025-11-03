import re

from typing import Optional, List
from pyquery import PyQuery as pq
from lxml.etree import Comment

from jet.scrapers.utils import BaseNode, TreeNode, extract_tree_with_text, get_leaf_nodes


def extract_nodes(element, depth: int = 0, parent_id: Optional[str] = None) -> List[BaseNode]:
    nodes = []
    element_pq = pq(element)
    text = element_pq.text().strip() if element_pq.text() else ""
    tag = element_pq[0].tag if element_pq else "unknown"
    # Skip comment nodes
    if tag is Comment or str(tag).startswith('<cyfunction Comment'):
        return nodes
    tag = tag.lower() if isinstance(tag, str) else "unknown"
    if tag == "html":
        for child in element_pq.children():
            nodes.extend(extract_nodes(child, depth + 1, parent_id))
        return nodes
    valid_id_pattern = r'^[a-zA-Z0-9_-]+$'
    element_id = element_pq.attr('id')
    element_class = element_pq.attr('class')
    adjusted_depth = max(1, depth - 2) if depth > 2 else 1
    id = element_id if element_id and re.match(
        valid_id_pattern, element_id) else f"node_{adjusted_depth}_{len(nodes)}"
    class_names = [name for name in (element_class.split() if element_class else [])
                   if re.match(valid_id_pattern, name)]
    if text and len(element_pq.children()) == 0:
        nodes.append(BaseNode(
            tag=tag,
            text=text,
            depth=adjusted_depth,
            raw_depth=depth,
            id=id,
            class_names=class_names,
            line=element_pq[0].sourceline if element_pq else 0,
            html=element_pq.outer_html()
        ))
    for child in element_pq.children():
        nodes.extend(extract_nodes(child, depth + 1, id))
    return nodes


def extract_text_nodes(
    source: str,
    excludes: List[str] = ["nav", "footer", "script", "style"],
    timeout_ms: int = 10000,
    with_screenshot: bool = True,
    wait_for_js: bool = True,
    headless: bool = True,
) -> List[TreeNode]:
    """
    Extracts text nodes from an HTML source using Playwright, excluding specified tags.
    Args:
        source: The HTML string, file path, or URL to parse.
        excludes: A list of tag names to exclude (e.g., ["nav", "footer", "script", "style"]).
        timeout_ms: Timeout for rendering the page (in ms) for dynamic content.
        with_screenshot: Whether to capture screenshots during rendering.
        wait_for_js: Whether to wait for JavaScript execution to complete.
        headless: Whether to run browser in headless mode.
    Returns:
        A list of TreeNode objects containing text, tag, depth, id, class_names, line number, and HTML.
    """
    # Get the tree-like structure
    tree_elements = extract_tree_with_text(
        source=source,
        excludes=excludes,
        timeout_ms=timeout_ms,
        with_screenshot=with_screenshot,
        wait_for_js=wait_for_js,
        headless=headless,
    )

    # Get leaf nodes with text
    leaf_nodes = get_leaf_nodes(tree_elements, with_text=True)
    
    return leaf_nodes