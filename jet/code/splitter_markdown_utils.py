import re
from typing import Callable, Optional, List, Dict, Tuple, TypedDict, Union

from jet.scrapers.preprocessor import html_to_markdown, is_html, scrape_markdown
from jet.scrapers.utils import clean_spaces
from jet.vectors.document_types import HeaderDocument
from .helpers.markdown_header_text_splitter import MarkdownHeaderTextSplitter


class HeaderMetadata(TypedDict):
    start_line_idx: int
    end_line_idx: int
    depth: int


class HeaderNode(TypedDict, total=False):
    header: str
    details: str
    content: str
    text: str
    metadata: HeaderMetadata
    is_root: bool
    child_nodes: List["HeaderNode"]


class HeaderItem(TypedDict):
    level: int
    header: str
    parents: list[str]


class HeaderLink(TypedDict):
    text: str
    url: str
    caption: Optional[str]
    start_idx: int
    end_idx: int
    line: str
    line_idx: int


class Header(TypedDict):
    header: str
    parent_header: Optional[str]
    header_level: int
    content: str
    text: str
    links: List[HeaderLink]


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
    """Extract the header text from a markdown or HTML header tag, including markdown hashes."""
    header = header.strip().splitlines()[0].strip()
    if header.startswith("#"):
        # Keep the leading hashes (up to the first space)
        parts = header.split(" ", 1)
        if len(parts) == 2:
            return f"{parts[0]} {parts[1].strip()}"
        return header
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
        return 0


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


def extract_markdown_links(text: str) -> Tuple[List[HeaderLink], str]:
    pattern = re.compile(r'\[([^\]]*)\]\((\S+?)(?:\s+"([^"]+)")?\)')
    links: List[HeaderLink] = []
    output = ""
    last_end = 0
    seen = set()
    all_lines = text.splitlines()

    for match in pattern.finditer(text):
        start, end = match.span()
        label, url, caption = match.group(1), match.group(2), match.group(3)

        # Append text before this match
        output += text[last_end:start]

        start_idx = len(output)
        replacement_text = label if label.strip() else ""
        output += replacement_text
        end_idx = len(output)

        # Find the full line containing the link and its line index
        start_line_idx = text[:start].rfind('\n') + 1
        end_line_idx = text.find('\n', end)
        if end_line_idx == -1:
            end_line_idx = len(text)
        line = text[start_line_idx:end_line_idx].strip()
        # Calculate line index (0-based)
        line_idx = len(text[:start].splitlines()) - 1

        # Uniqueness key ignoring positions
        key = (label, url, caption, line)
        if key not in seen:
            seen.add(key)
            links.append({
                "text": label,
                "url": url,
                "caption": caption,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "line": line,
                "line_idx": line_idx
            })

        last_end = end

    # Append remaining text after last match
    output += text[last_end:]

    return links, output


def get_md_header_contents(
    md_text: str,
    headers_to_split_on: List[Tuple[str, str]] = [],
    ignore_links: bool = True
) -> List[Header]:
    from jet.scrapers.utils import clean_newlines, clean_text, clean_spaces
    from jet.scrapers.preprocessor import is_html, html_to_markdown

    if is_html(md_text):
        md_text = html_to_markdown(md_text, ignore_links=ignore_links, remove_selectors=[
            "style", "script", "nav", "footer"])

    md_text = md_text.strip()

    headers_to_split_on = headers_to_split_on or [
        ("#", "h1"), ("##", "h2"), ("###", "h3"),
        ("####", "h4"), ("#####", "h5"), ("######", "h6"),
    ]
    headers_to_split_on += [(f"* {header}", tag)
                            for header, tag in headers_to_split_on]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on, strip_headers=False, return_each_line=False
    )
    md_header_splits = markdown_splitter.split_text(md_text)

    md_header_contents: List[Header] = []
    parent_stack: List[Tuple[int, str]] = []

    for split in md_header_splits:
        raw_text = clean_spaces(clean_newlines(clean_text(
            split.page_content), max_newlines=1, strip_lines=True))

        # Check if metadata is empty
        if not split.metadata:
            # Skip splits with no metadata or handle as plain content
            if raw_text.strip():
                if md_header_contents and parent_stack:
                    # Append to the last header's content if available
                    md_header_contents[-1]["content"] += "\n\n" + raw_text
                    md_header_contents[-1]["text"] += "\n\n" + raw_text
                continue

        last_key = list(split.metadata.keys())[-1]
        last_value = split.metadata[last_key]
        last_hashtag = next(
            (prefix for prefix, header in headers_to_split_on if header == last_key), None)
        last_header = f"{last_hashtag} {last_value}"

        if not raw_text.startswith(last_header):
            for key, value in split.metadata.items():
                hashtag = next(
                    (prefix for prefix, header in headers_to_split_on if header == key), None)
                header = f"{hashtag} {value}"
                if raw_text.startswith(header):
                    raw_text = raw_text[len(header):].strip()
            raw_text = f"{last_header}\n{raw_text}"

        try:
            # Extract raw header line and clean it
            header_line = get_header_text(raw_text)
            header_line = clean_spaces(header_line)

            # Extract links from header line
            header_links, clean_header = extract_markdown_links(header_line)

            header_level = get_header_level(clean_header)

            # Clean full text, extract links
            cleaned_text = clean_spaces(raw_text)
            body_links, cleaned_text = extract_markdown_links(cleaned_text)

            # Combine all links
            all_links = header_links + body_links

            # Remove duplicates based on (text, url, caption, line)
            seen = set()
            unique_links = []
            for link in all_links:
                key = (link["text"], link["url"],
                       link["caption"], link["line"])
                if key not in seen:
                    seen.add(key)
                    unique_links.append(link)

            all_links = unique_links

            # Extract content below the header
            content_lines = cleaned_text.splitlines()[1:]
            content = "\n".join(
                line for line in content_lines
                if not any(line.lstrip().startswith(prefix) for prefix, _ in headers_to_split_on)
            ).strip()
            clean_content = clean_spaces(content)

            final_text = f"{clean_header}\n{clean_content}".strip()
            header_text = clean_header.lstrip("#").strip()

            if not header_text and parent_stack:
                if md_header_contents:
                    md_header_contents[-1]["content"] += "\n\n" + clean_content
                    md_header_contents[-1]["text"] += "\n\n" + final_text
                continue

            while parent_stack and parent_stack[-1][0] >= header_level:
                parent_stack.pop()
            parent_header = parent_stack[-1][1] if parent_stack else None
            parent_stack.append((header_level, clean_header))

            md_header_contents.append({
                "header": clean_header,
                "header_level": header_level,
                "parent_header": parent_header,
                "content": clean_content,
                "text": final_text,
                "links": all_links,
            })
        except ValueError:
            continue

    return md_header_contents


def get_md_header_docs(
    md_text: str,
    headers_to_split_on: List[tuple[str, str]] = [],
    ignore_links: bool = False
) -> List[HeaderDocument]:
    headers = get_md_header_contents(
        md_text, headers_to_split_on, ignore_links)
    header_docs = [
        HeaderDocument(
            doc_index=i,
            **header,
        )
        for i, header in enumerate(headers)
    ]

    return header_docs


def merge_md_header_contents(
    header_contents: list[dict],
    min_tokens: int = 256,
    max_tokens: int = 1000,
    tokenizer: Optional[Callable[[str], List]] = None
) -> list[dict]:
    merged_header_contents = []
    merged_content = ""

    # Validate that min_tokens is less than max_tokens
    if min_tokens > max_tokens:
        raise ValueError(
            f"min_tokens ({min_tokens}) cannot be greater than max_tokens ({max_tokens}).")

    extracted_contents = [header_content["content"]
                          for header_content in header_contents]
    all_header_stack: list[HeaderItem] = []
    last_headers = []

    for content in extracted_contents:
        content_len = count_tokens(content, tokenizer)
        merged_content_len = count_tokens(merged_content, tokenizer)
        header_level = get_header_level(content)
        header_line = content.splitlines()[0] if content else ""

        # Ensure the current chunk isn't below min_tokens unless unavoidable
        if merged_content and merged_content_len + content_len > max_tokens:
            if merged_content_len < min_tokens and merged_content_len + content_len <= max_tokens:
                # Allow adding more content if it helps reach min_tokens
                pass
            else:
                header = merged_content.splitlines()[0]
                parent_headers = [
                    last_header.strip() for last_header in last_headers
                    if get_header_level(last_header) < get_header_level(header)
                ]
                merged_header_contents.append({
                    "header": header,
                    "length": merged_content_len,
                    "content": merged_content,
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
            item['header']: item['parents'] for item in all_header_stack
        }

    # Append remaining content if it exists
    if merged_content:
        header = merged_content.splitlines()[0]
        parent_headers = [
            last_header.strip() for last_header in last_headers
            if get_header_level(last_header) < get_header_level(header)
        ]
        merged_header_contents.append({
            "header": header,
            "length": count_tokens(merged_content, tokenizer),
            "content": merged_content,
            "parent_headers": parent_headers,
        })

    return merged_header_contents


def extract_md_header_contents(md_text: str, min_tokens_per_chunk: int = 256, max_tokens_per_chunk: int = 1000, tokenizer: Optional[Callable[[str], List]] = None) -> list[dict]:
    header_contents = get_md_header_contents(md_text)
    header_contents = merge_md_header_contents(
        header_contents, min_tokens=min_tokens_per_chunk, max_tokens=max_tokens_per_chunk, tokenizer=tokenizer)

    # Clean newlines and extra spaces
    for header_content in header_contents:
        header_content["header_level"] = get_header_level(
            header_content["header"])

    return header_contents


def count_md_header_contents(md_text: str, headers_to_split_on: list[tuple[str, str]] = []) -> int:
    header_contents = get_md_header_contents(md_text, headers_to_split_on)
    return len(header_contents)


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
