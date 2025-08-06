import os
import re
from typing import Optional, List
from jet.scrapers.utils import BaseNode, exclude_elements
from pyquery import PyQuery as pq
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError


def extract_text_nodes(
    source: str,
    excludes: List[str] = ["nav", "footer", "script", "style"],
    timeout_ms: int = 1000
) -> List[BaseNode]:
    """
    Extracts a list of BaseNode objects containing text from the HTML document, ignoring specific elements.
    Uses Playwright to render dynamic content if needed.

    :param source: The HTML string or URL to parse.
    :param excludes: A list of tag names to exclude (e.g., ["nav", "footer", "script", "style"]).
    :param timeout_ms: Timeout for rendering the page (in ms) for dynamic content.
    :return: A list of BaseNode objects representing text-containing elements in the HTML.
    """
    if os.path.exists(source) and not source.startswith("file://"):
        source = f"file://{source}"

    if re.match(r'^https?://', source) or re.match(r'^file://', source):
        url = source
        html = None
    else:
        url = None
        html = source

    # Use Playwright to render the page if URL is provided
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        if url:
            page.goto(url, wait_until="networkidle")
        else:
            page.set_content(html)

        page.wait_for_timeout(timeout_ms)

        # Extract the content
        page_content = page.content()
        browser.close()

    # Parse the content with PyQuery after Playwright has rendered it
    doc = pq(page_content)

    # Apply the exclusion logic before extracting nodes
    exclude_elements(doc, excludes)

    def extract_nodes(element, depth: int = 0, parent_id: Optional[str] = None) -> List[BaseNode]:
        nodes = []
        element_pq = pq(element)
        text = element_pq.text().strip()
        tag = element_pq[0].tag if element_pq else "unknown"

        valid_id_pattern = r'^[a-zA-Z_-]+$'
        element_id = element_pq.attr('id')
        element_class = element_pq.attr('class')
        id = element_id if element_id and re.match(
            valid_id_pattern, element_id) else f"node_{depth}_{len(nodes)}"
        class_names = [name for name in (element_class.split() if element_class else [])
                       if re.match(valid_id_pattern, name)]
        link = element_pq.attr('href') if tag == 'a' else None

        # Create a BaseNode if the element has text and no children
        if text and len(element_pq.children()) == 0:
            nodes.append(BaseNode(
                tag=tag,
                text=text,
                depth=depth,
                id=id,
                class_names=class_names,
                line=element_pq[0].sourceline if element_pq else 0,
                html=element_pq.outer_html()
            ))

        # Recursively process children
        for child in element_pq.children():
            nodes.extend(extract_nodes(child, depth + 1, id))

        return nodes

    # Start with the root element and gather all text nodes
    text_nodes = extract_nodes(doc[0])

    return text_nodes
