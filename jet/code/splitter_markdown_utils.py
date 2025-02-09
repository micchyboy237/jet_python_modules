from typing import Optional, List, Dict, TypedDict, Union


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

    hierarchy = build_hierarchy(header_nodes)

    if include_child_contents:
        for node in header_nodes:
            full_content = collect_full_content(node)
            full_content_lines = full_content.splitlines()
            start_line_idx = all_lines.index(full_content_lines[0])
            end_line_idx = all_lines.index(full_content_lines[-1]) + 1
            node["content"] = full_content
            node["metadata"]["start_line_idx"] = start_line_idx
            node["metadata"]["end_line_idx"] = end_line_idx

    return hierarchy
