from typing import Optional, List, TypedDict
from jet.data.utils import generate_unique_id
from jet.scrapers.text_nodes import extract_text_nodes
from jet.scrapers.utils import ElementDetails


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
    element: Optional[ElementDetails]
    # html: str


def extract_header_hierarchy(
    source: str,
    excludes: List[str] = ["nav", "footer", "script", "style"],
    timeout_ms: int = 10000
) -> List[HtmlHeaderDoc]:
    """
    Extracts a list of HtmlHeaderDoc objects from HTML content, organizing text by header hierarchy.
    Ignores content before the first header.
    :param source: The HTML string or URL to parse.
    :param excludes: A list of tag names to exclude (e.g., ["nav", "footer", "script", "style"]).
    :param timeout_ms: Timeout for rendering the page (in ms) for dynamic content.
    :return: A list of HtmlHeaderDoc objects representing header-based sections.
    """
    nodes = extract_text_nodes(source, excludes, timeout_ms)
    sections: List[HtmlHeaderDoc] = []
    current_section: Optional[HtmlHeaderDoc] = None
    header_stack: List[tuple[str, int, int]] = []
    current_content: List[str] = []
    current_html_content: List[str] = []
    section_index = 0
    header_tags = {f'h{i}': i for i in range(1, 7)}
    base_depth: Optional[int] = None  # Store the raw_depth of the first header

    for node in nodes:
        # if node.tag == "label" and node.is_clickable:
        #     continue

        tag = node.tag.lower()
        text = node.text.strip() if node.text else ""
        if tag in header_tags and text:
            # Save current section if it has a header and content
            if current_section and current_section["header"].strip():
                current_section["content"] = "\n".join(current_content)
                # current_section["html"] = "\n".join(current_html_content)
                sections.append(current_section)
                section_index += 1
            current_content = []
            current_html_content = []
            level = header_tags[tag]
            # Set base_depth to the first header's depth
            if base_depth is None:
                base_depth = node.depth
            # Calculate depth relative to the first header's depth
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
                "element": node.get_element_details(),
                # "html": node.get_html(),
            }
            header_stack.append((text, level, section_index))
            current_html_content.append(node.get_html())
        else:
            if text and current_section is not None:
                current_content.append(text)
                current_html_content.append(node.get_html())

    # Append the final section if it has a header and content
    if current_section and current_section["header"].strip():
        current_section["content"] = "\n".join(current_content)
        current_section["html"] = "\n".join(current_html_content)
        sections.append(current_section)

    # Filter out empty sections and reindex
    sections = [
        section for section in sections
        if section["header"].strip() or section["content"].strip()
    ]
    for idx, section in enumerate(sections):
        section["doc_index"] = idx

    return sections
