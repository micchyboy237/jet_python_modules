import os
import re
from typing import Optional, List
from jet.scrapers.utils import BaseNode, exclude_elements
from pyquery import PyQuery as pq
from lxml.etree import Comment
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError


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
    timeout_ms: int = 1000
) -> List[BaseNode]:
    """
    Extracts text nodes from an HTML source, excluding specified tags.
    Args:
        source: The HTML string, file path, or URL to parse.
        excludes: A list of tag names to exclude (e.g., ["nav", "footer", "script", "style"]).
        timeout_ms: Timeout for rendering the page (in ms) for dynamic content.
    Returns:
        A list of BaseNode objects containing text, tag, depth, id, class_names, line number, and HTML.
    """
    if os.path.exists(source) and not source.startswith("file://"):
        source = f"file://{source}"
    if re.match(r'^https?://', source) or re.match(r'^file://', source):
        url = source
        html = None
    else:
        url = None
        html = source
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        if url:
            page.goto(url, wait_until="networkidle")
        else:
            page.set_content(html)
        page.wait_for_timeout(timeout_ms)
        page_content = page.content()
        browser.close()
    doc = pq(page_content)
    for tag in excludes:
        doc(tag).remove()
    return extract_nodes(doc[0])
