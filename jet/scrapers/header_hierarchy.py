from typing import Optional, List, TypedDict
from jet.data.utils import generate_unique_id
from jet.scrapers.text_nodes import extract_text_nodes


class HeaderDoc(TypedDict):
    id: str
    doc_index: int
    tag: str
    parent_level: Optional[int]
    level: Optional[int]
    parent_headers: List[str]
    parent_header: Optional[str]
    header: str
    content: str
    html: str


def extract_header_hierarchy(
    source: str,
    excludes: List[str] = ["nav", "footer", "script", "style"],
    timeout_ms: int = 1000
) -> List[HeaderDoc]:
    """
    Extracts a list of HeaderDoc objects from HTML content, organizing text by header hierarchy.

    :param source: The HTML string or URL to parse.
    :param excludes: A list of tag names to exclude (e.g., ["nav", "footer", "script", "style"]).
    :param timeout_ms: Timeout for rendering the page (in ms) for dynamic content.
    :return: A list of HeaderDoc objects representing header-based sections.
    """
    nodes = extract_text_nodes(source, excludes, timeout_ms)

    sections: List[HeaderDoc] = []
    current_section: Optional[HeaderDoc] = None
    header_stack: List[tuple[str, int, int]] = []
    current_content: List[str] = []
    section_index = 0

    header_tags = {f'h{i}': i for i in range(1, 7)}

    for node in nodes:
        tag = node.tag.lower()
        text = node.text.strip()

        if tag in header_tags and text:
            if current_section:
                current_section["content"] = "\n".join(current_content)
                sections.append(current_section)
                section_index += 1
                current_content = []

            level = header_tags[tag]
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
                "parent_level": parent_level,
                "level": level,
                "parent_headers": parent_headers,
                "parent_header": parent_header,
                "header": text,
                "content": "",
                "html": node.get_html()
            }
            header_stack.append((text, level, section_index))
        else:
            if text:
                if current_section is None:
                    current_section = {
                        "id": generate_unique_id(),
                        "doc_index": section_index,
                        "tag": "",
                        "parent_header": None,
                        "header": "",
                        "content": "",
                        "level": 0,
                        "parent_headers": [],
                        "parent_level": None,
                        "html": ""
                    }
                current_content.append(text)

    if current_section:
        current_section["content"] = "\n".join(current_content)
        if current_section["header"].strip() or current_section["content"].strip():
            sections.append(current_section)

    sections = [
        section for section in sections
        if section["header"].strip() or section["content"].strip()
    ]
    for idx, section in enumerate(sections):
        section["doc_index"] = idx

    return sections
