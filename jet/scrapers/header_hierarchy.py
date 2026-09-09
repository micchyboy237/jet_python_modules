from typing import List, Optional, TypedDict

from jet.data.utils import generate_unique_id
from jet.scrapers.text_nodes import extract_text_nodes
from jet.scrapers.utils import ElementDetails
from jet.scrapers.utils.scrape_links import scrape_links
from lxml.etree import tostring


class HtmlHeaderDoc(TypedDict):
    id: str
    doc_index: int
    tag: str
    depth: int
    parent_level: Optional[int]
    level: Optional[int]
    parent_headers: List[str]
    parent_header: Optional[str]
    header: str
    content: str
    html: str
    element: Optional[ElementDetails]
    links: List[str]  # URLs found within this header's section


# Block-level tags whose outer HTML should be captured as structural fragments.
_BLOCK_TAGS = frozenset(
    {
        "table",
        "thead",
        "tbody",
        "tfoot",
        "tr",
        "td",
        "th",
        "pre",
        "code",
        "ul",
        "ol",
        "li",
        "dl",
        "dt",
        "dd",
        "blockquote",
        "figure",
        "figcaption",
        "details",
        "summary",
        "div",
        "section",
        "article",
        "aside",
        "main",
        "p",
    }
)


def _get_block_ancestor_html(node) -> str:
    """
    Walk up from a leaf text node to find the nearest meaningful block ancestor
    and return its outer HTML. Falls back to the leaf node's own HTML if no
    block ancestor is found.
    """
    element = node.get_element()
    if element is None:
        return node.get_html() or ""

    current = element
    while current is not None:
        tag = current.tag
        if isinstance(tag, str):
            tag_lower = tag.lower()
        else:
            tag_lower = str(tag).lower()

        if tag_lower in _BLOCK_TAGS:
            try:
                return tostring(current, encoding="unicode", method="html")
            except Exception:
                pass

        parent = current.getparent()
        if parent is None:
            break
        current = parent

    return node.get_html() or ""


def extract_header_hierarchy(
    source: str,
    includes: List[str] = [],
    excludes: List[str] = ["nav", "footer", "script", "style"],
    timeout_ms: int = 10000,
    ignore_links: bool = False,
) -> List[HtmlHeaderDoc]:
    """
    Extracts a list of HtmlHeaderDoc objects from HTML content, organizing text by header hierarchy.
    Each section's ``html`` field contains the outer HTML of all block-level ancestors
    for the leaf text nodes within that section.
    Each section's ``links`` field contains unique URLs found within that section.
    """
    header_tags = {f"h{i}": i for i in range(1, 7)}
    nodes = extract_text_nodes(
        source,
        excludes=excludes,
        timeout_ms=timeout_ms,
    )
    if includes:
        nodes = [
            node for node in nodes if node.tag in includes + list(header_tags.keys())
        ]
    sections: List[HtmlHeaderDoc] = []
    current_section: Optional[HtmlHeaderDoc] = None
    header_stack: List[tuple[str, int, int]] = []
    current_content: List[str] = []
    current_html_content: List[str] = []
    seen_block_html: set = set()
    seen_links: set = set()  # NEW: Track unique links per section
    section_index = 0
    base_depth: Optional[int] = None

    for node in nodes:
        tag = node.tag.lower()
        text = node.text.strip() if node.text else ""

        if tag in header_tags and text:
            # Finalize previous section
            if current_section and current_section["header"].strip():
                current_section["content"] = "\n".join(current_content)
                current_section["html"] = "\n".join(current_html_content)
                sections.append(current_section)
                section_index += 1
                current_content = []
                current_html_content = []
                seen_block_html = set()
                seen_links = set()  # NEW: Reset link tracker

            level = header_tags[tag]
            if base_depth is None:
                base_depth = node.depth
            depth = max(1, node.depth - base_depth + 1)

            parent_headers = []
            parent_header = None
            parent_level = None
            while header_stack and header_stack[-1][1] >= level:
                header_stack.pop()
            if header_stack:
                parent_header = header_stack[-1][0]
                parent_level = header_stack[-1][1]
                parent_headers = [h[0] for h in header_stack]

            current_section = {
                "id": generate_unique_id(),
                "doc_index": section_index,
                "tag": tag,
                "depth": depth,
                "parent_level": parent_level,
                "level": level,
                "parent_headers": parent_headers,
                "parent_header": parent_header,
                "header": text,
                "content": "",
                "html": "",
                "element": node.get_element_details(),
                "links": [],  # NEW: Initialize links list
            }
            header_stack.append((text, level, section_index))

            heading_html = node.get_html() or ""
            if heading_html:
                current_html_content.append(heading_html)
                seen_block_html.add(heading_html)

                # NEW: Extract links from header HTML
                if not ignore_links:
                    header_links = scrape_links(heading_html)
                    for link in header_links:
                        if link not in seen_links:
                            seen_links.add(link)
                            current_section["links"].append(link)
        else:
            if text and current_section is not None:
                current_content.append(text)
                block_html = _get_block_ancestor_html(node)
                if block_html and block_html not in seen_block_html:
                    current_html_content.append(block_html)
                    seen_block_html.add(block_html)

                    # NEW: Extract links from content block HTML
                    if not ignore_links:
                        block_links = scrape_links(block_html)
                        for link in block_links:
                            if link not in seen_links:
                                seen_links.add(link)
                                current_section["links"].append(link)

    # Finalize last section
    if current_section and current_section["header"].strip():
        current_section["content"] = "\n".join(current_content)
        current_section["html"] = "\n".join(current_html_content)
        sections.append(current_section)

    sections = [
        section
        for section in sections
        if section["header"].strip() or section["content"].strip()
    ]
    for idx, section in enumerate(sections):
        section["doc_index"] = idx

    return sections
