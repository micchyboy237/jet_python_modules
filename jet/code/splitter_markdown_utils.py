import re
from typing import Callable, Optional, List, Dict, TypedDict, Union

from jet.scrapers.preprocessor import scrape_markdown


class HeaderMetadata(TypedDict):
    start_line_idx: int
    end_line_idx: int
    depth: int


class HeaderNode(TypedDict, total=False):
    header: str
    details: str
    content: str
    metadata: HeaderMetadata
    is_root: bool
    child_nodes: List["HeaderNode"]


class HeaderItem(TypedDict):
    level: int
    header: str
    parents: list[str]


def get_flat_header_list(header_nodes: Union[HeaderNode, List[HeaderNode]], flat_list: Optional[List[HeaderNode]] = None) -> List[HeaderNode]:
    """Returns a flat list of header nodes, including itself and its children. Can handle a single header node or a list of header nodes."""
    if flat_list is None:
        flat_list = []

    if isinstance(header_nodes, list):
        # If input is a list of HeaderNode, process each node in the list
        for node in header_nodes:
            get_flat_header_list(node, flat_list)
    else:
        # If input is a single HeaderNode, process it
        flat_list.append(header_nodes)
        for child in header_nodes.get("child_nodes", []):
            get_flat_header_list(child, flat_list)

    return flat_list


def get_header_text(header: str) -> str:
    """Extract the header text from a markdown or HTML header tag."""
    header = header.splitlines()[0]
    header = header.strip()
    if header.startswith("#"):
        return header.lstrip("#").strip()
    elif header.startswith("h") and header[1].isdigit() and 1 <= int(header[1]) <= 6:
        return header[2:].strip()
    else:
        raise ValueError(f"Invalid header format: {header}")


def get_header_level(header: str) -> int:
    """Get the header level of a markdown header or HTML header tag."""
    header = header.splitlines()[0]
    header = header.strip()
    if header.startswith("#"):
        header_level = 0
        for c in header:
            if c == "#":
                header_level += 1
            else:
                break
        return header_level
    elif header.startswith("h") and header[1].isdigit() and 1 <= int(header[1]) <= 6:
        return int(header[1])
    else:
        raise ValueError(f"Invalid header format: {header}")


def build_nodes_hierarchy(headers: List[HeaderNode]) -> List[HeaderNode]:
    """Convert flat header list to a nested hierarchy based on depth."""
    stack: List[HeaderNode] = []

    for header in headers:
        while stack and stack[-1]["metadata"]["depth"] >= header["metadata"]["depth"]:
            stack.pop()

        if stack:
            parent = stack[-1]
            if "child_nodes" not in parent:
                parent["child_nodes"] = []
            parent["child_nodes"].append(header)
        else:
            header["is_root"] = True

        stack.append(header)

    return [header for header in headers if header.get("is_root")]


def collect_nodes_full_content(node: HeaderNode) -> str:
    """Recursively collect content from a node and its children."""
    content = node["content"]
    if "child_nodes" in node:
        child_content = "\n".join(collect_nodes_full_content(child)
                                  for child in node["child_nodes"])
        content += child_content
    return content.rstrip()


def get_header_contents(md_text: str,
                        headers_to_split_on: Optional[List[str]] = None,
                        include_child_contents: bool = True) -> List[HeaderNode]:
    header_lines = []
    headers_to_split_on = headers_to_split_on or [
        "#", "##", "###", "####", "#####", "######"]

    header_prefixes = [f"{prefix.strip()} " for prefix in headers_to_split_on]
    all_lines = md_text.splitlines()

    for line_idx, line in enumerate(all_lines):
        if any(line.lstrip().startswith(prefix) for prefix in header_prefixes):
            header_lines.append({"index": line_idx, "line": line})

    header_content_indexes = [item["index"]
                              for item in header_lines] + [len(all_lines)]
    header_content_ranges = [
        (header_content_indexes[item_idx],
         header_content_indexes[item_idx + 1])
        for item_idx, _ in enumerate(header_lines)
    ]

    header_nodes: List[HeaderNode] = []

    for start_idx, end_idx in header_content_ranges:
        header_line, *contents = all_lines[start_idx:end_idx]
        header_level = get_header_level(header_line)
        content = "\n".join(contents)

        # details = content if content.strip() else "<placeholder>"
        details = content if content.strip() else ""

        block_content = f"{header_line}\n{details}"

        header_nodes.append({
            "header": header_line,
            "details": details,
            "content": block_content,
            "metadata": {
                "start_line_idx": start_idx,
                "end_line_idx": end_idx,
                "depth": header_level,
            },
        })

    hierarchy = build_nodes_hierarchy(header_nodes)

    if include_child_contents:
        for node in header_nodes:
            full_content = collect_nodes_full_content(node)
            full_content_lines = full_content.splitlines()
            start_line_idx = all_lines.index(full_content_lines[0])
            end_line_idx = all_lines.index(full_content_lines[-1]) + 1
            node["content"] = full_content
            node["metadata"]["start_line_idx"] = start_line_idx
            node["metadata"]["end_line_idx"] = end_line_idx

    return hierarchy


def get_md_header_contents(md_text: str, headers_to_split_on: list[tuple[str, str]] = []) -> list[dict]:
    from langchain_text_splitters import MarkdownHeaderTextSplitter

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on, strip_headers=False, return_each_line=False)
    md_header_splits = markdown_splitter.split_text(md_text)
    md_header_contents = []
    for split in md_header_splits:
        content = split.page_content
        # metadata = split.metadata

        if content.strip():
            try:
                md_header_contents.append({
                    "content": content.strip(),
                    "length": len(content.strip()),
                    "header": get_header_text(content),
                    "header_level": get_header_level(content),
                })
            except ValueError:
                continue
    return md_header_contents


def merge_md_header_contents(header_contents: list[dict], max_tokens: int = 1000, tokenizer: Optional[Callable[[str], List]] = None) -> list[dict]:
    merged_header_contents = []
    merged_content = ""

    extracted_contents = [header_content["content"]
                          for header_content in header_contents]

    all_header_stack: list[HeaderItem] = []
    last_headers = []

    for content in extracted_contents:
        content_len = count_tokens(content, tokenizer)
        merged_content_len = count_tokens(merged_content, tokenizer)
        header_level = get_header_level(content)
        header_line = content.splitlines()[0] if content else ""

        if merged_content and merged_content_len + content_len > max_tokens:
            header = merged_content.splitlines()[0]
            parent_headers = [last_header.strip() for last_header in last_headers
                              if get_header_level(last_header) < get_header_level(header)]
            merged_header_contents.append({
                "content": merged_content,
                "length": count_tokens(merged_content, tokenizer),
                "header": header,
                "parent_headers": parent_headers,
            })
            merged_content = ""

            last_header_line = list(all_header_stack_dict.keys())[-1]
            last_headers = all_header_stack_dict[last_header_line]

        if merged_content:
            merged_content += "\n"  # Ensure newline between merged contents
        merged_content += content

        all_header_stack.append({"level": header_level, "header": header_line})
        all_header_stack = add_parents(all_header_stack)
        all_header_stack_dict: dict[str, list[str]] = {
            item['header']: item['parents'] for item in all_header_stack}

    if merged_content:
        header = merged_content.splitlines()[0]
        parent_headers = [last_header.strip() for last_header in last_headers
                          if get_header_level(last_header) < get_header_level(header)]
        merged_header_contents.append({
            "content": merged_content,
            "length": count_tokens(merged_content, tokenizer),
            "header": merged_content.splitlines()[0],
            "parent_headers": parent_headers,
        })

    return merged_header_contents


def extract_md_header_contents(md_text: str, max_tokens_per_chunk: int = 1000, tokenizer: Optional[Callable[[str], List]] = None) -> list[dict]:
    headers_to_split_on = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
        ("####", "h4"),
        ("#####", "h5"),
        ("######", "h6"),
    ]

    header_contents = get_md_header_contents(md_text, headers_to_split_on)
    header_contents = merge_md_header_contents(
        header_contents, max_tokens=max_tokens_per_chunk, tokenizer=tokenizer)

    # Clean newlines and extra spaces
    for header_content in header_contents:
        header_content["content"] = clean_newlines(header_content["content"])

    return header_contents


def extract_html_header_contents(html_str: str) -> list[dict]:
    headers_to_split_on = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
        ("####", "h4"),
        ("#####", "h5"),
        ("######", "h6"),
    ]

    scraped_result = scrape_markdown(html_str)
    header_contents = get_md_header_contents(
        scraped_result['content'], headers_to_split_on)

    return header_contents


def clean_newlines(content):
    """Remove consecutive newlines from the content."""
    # Remove trailing whitespace for each line
    content = '\n'.join([line.rstrip() for line in content.split('\n')])
    # Reduce consecutive newlines to a single newline
    content = re.sub(r'\n+', '\n', content)
    return content


def count_tokens(text: str, tokenizer: Optional[Callable[[str], List]] = None) -> int:
    if tokenizer:
        count = len(tokenizer(text))
    else:
        count = len(text)
    return count


def add_parents(items: list[HeaderItem]) -> list[HeaderItem]:
    hierarchy = []  # Stack to keep track of parent headers

    for item in items:
        level = item["level"]

        # Remove items from hierarchy that are no longer parents
        while hierarchy and hierarchy[-1]["level"] >= level:
            hierarchy.pop()

        # Assign parents based on remaining hierarchy
        item["parents"] = [parent["header"] for parent in hierarchy]

        # Add the current item to hierarchy stack
        hierarchy.append(item)

    return items


__all__ = [
    "get_flat_header_list",
    "get_header_level",
    "build_nodes_hierarchy",
    "collect_nodes_full_content",
    "get_header_contents",
    "get_md_header_contents",
    "merge_md_header_contents",
    "extract_md_header_contents",
]
