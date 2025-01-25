from typing import Optional, List, Dict, TypedDict


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


def get_header_level(header_line: str) -> int:
    """Get the depth level of a markdown header."""
    return header_line.count('#')


def build_hierarchy(headers: List[HeaderNode]) -> List[HeaderNode]:
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


def collect_full_content(node: HeaderNode) -> str:
    """Recursively collect content from a node and its children."""
    content = node["content"]
    if "child_nodes" in node:
        child_content = "\n".join(collect_full_content(child)
                                  for child in node["child_nodes"])
        content += "\n" + child_content
    return content


def get_header_contents(md_text: str,
                        headers_to_split_on: Optional[List[str]] = None,
                        include_child_contents: bool = False) -> List[HeaderNode]:
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

        details = content if content.strip() else "<placeholder>"
        block_content = f"{header_line}\n\n{details}\n\n"

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

    hierarchy = build_hierarchy(header_nodes)

    if include_child_contents:
        for node in header_nodes:
            node["content"] = collect_full_content(node)

    return hierarchy
