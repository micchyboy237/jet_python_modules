from urllib.parse import urljoin
import uuid
from jet.code.html_utils import is_html
from jet.llm.mlx.helpers.detect_repetition import clean_repeated_ngrams
from jet.logger import logger
from jet.code.markdown_types import MarkdownAnalysis
from jet.code.markdown_utils import analyze_markdown, convert_html_to_markdown
from typing import Any, List, Optional, Tuple
import re
from typing import Callable, Optional, List, Dict, Tuple, TypedDict, Union
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup

from jet.scrapers.preprocessor import html_to_markdown, scrape_markdown
from jet.scrapers.utils import clean_newlines, clean_text, clean_spaces
from jet.vectors.document_types import HeaderDocument, HeaderDocumentDict, HeaderMetadata as HeaderMetadataDoc
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
    is_heading: bool


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


def extract_markdown_links(text: str, base_url: Optional[str] = None, ignore_links: bool = True) -> Tuple[List[HeaderLink], str]:
    """
    Extracts markdown links and plain URLs from text, optionally replacing them with their text content or cleaning URLs.
    Handles nested image links like [![alt](image_url)](link_url) and reference-style links like [text][ref].

    Args:
        text: Input string containing markdown links or plain URLs.
        base_url: Base URL to resolve relative links, if provided.
        ignore_links: If True, replaces links with text content in output; if False, preserves links.

    Returns:
        Tuple of a list of HeaderLink dictionaries and the modified text.
    """
    # Pattern for markdown links, including nested image links
    pattern = re.compile(
        r'\[(?:!\[([^\]]*?)\]\(([^)]+?)(?:\s+"[^"]*")?\)|([^\]]*))\]\((\S+?)(?:\s+"([^"]*)")?\)|'
        # Capture reference-style links [text][ref]
        r'\[([^\]]*)\]\[([^\]]*)\]',
        re.MULTILINE
    )
    # Pattern for reference definitions [ref]: url
    ref_pattern = re.compile(
        r'^\[([^\]]*)\]:\s*(\S+)(?:\s+"([^"]*)")?$',
        re.MULTILINE
    )
    plain_url_pattern = re.compile(
        r'(?<!\]\()https?://[^\s<>\]\)]+[^\s<>\]\).,?!]',
        re.MULTILINE
    )
    links: List[HeaderLink] = []
    output = text
    seen: Set[Tuple[str, str, Optional[str], str]] = set()
    replacements: List[Tuple[int, int, str]] = []
    # Store reference URLs and captions
    ref_urls: Dict[str, Tuple[str, Optional[str]]] = {}

    # Extract reference definitions first
    for match in ref_pattern.finditer(text):
        ref_id = match.group(1).strip()
        url = match.group(2).strip()
        caption = match.group(3)
        if ref_id and url:
            ref_urls[ref_id.lower()] = (url, caption)

    # Extract markdown links
    for match in pattern.finditer(text):
        start, end = match.span()
        image_alt, image_url, label, url, caption, ref_text, ref_id = match.groups()
        selected_url = ""
        selected_caption = caption

        if ref_text and ref_id:  # Handle reference-style link [text][ref]
            label = ref_text
            if ref_id.lower() in ref_urls:
                selected_url, selected_caption = ref_urls[ref_id.lower()]
            else:
                continue  # Skip if reference not found
        else:
            label = image_alt if image_alt else label  # Use image alt as label if present
            # Prioritize outer link URL if present, otherwise use image URL
            selected_url = url.strip() if url and not image_url else (
                image_url.strip() if image_url else "")

        if not selected_url:  # Skip if no valid URL
            continue

        # Convert relative URLs to absolute
        if base_url and not selected_url.startswith(('http://', 'https://')):
            selected_url = urljoin(base_url, selected_url)

        # Find line and line index
        start_line_idx = text[:start].rfind('\n') + 1
        end_line_idx = text.find('\n', end)
        if end_line_idx == -1:
            end_line_idx = len(text)
        line = text[start_line_idx:end_line_idx].strip()
        line_idx = len(text[:start].splitlines()) - 1

        # Create link entry
        key = (label or "", selected_url, selected_caption, line)
        if key not in seen:
            seen.add(key)
            links.append({
                "text": label or "",
                "url": selected_url,
                "caption": selected_caption,
                "start_idx": start,
                "end_idx": end,
                "line": line,
                "line_idx": line_idx,
                "is_heading": line.startswith('#')
            })
        if ignore_links and label and label.strip():
            replacements.append((start, end, label))
        elif ignore_links:
            replacements.append((start, end, ""))
        else:
            replacements.append((start, end, match.group(0)))

    # Extract plain URLs (unchanged)
    for match in plain_url_pattern.finditer(text):
        url = match.group(0).strip()
        start, end = match.span()
        if not any(url in link["url"] for link in links):  # Avoid duplicates
            start_line_idx = text[:start].rfind('\n') + 1
            end_line_idx = text.find('\n', end)
            if end_line_idx == -1:
                end_line_idx = len(text)
            line = text[start_line_idx:end_line_idx].strip()
            line_idx = len(text[:start].splitlines()) - 1
            key = ("", url, None, line)
            if key not in seen:
                seen.add(key)
                links.append({
                    "text": "",
                    "url": url,
                    "caption": None,
                    "start_idx": start,
                    "end_idx": end,
                    "line": line,
                    "line_idx": line_idx,
                    "is_heading": line.startswith('#')
                })
            if ignore_links:
                replacements.append((start, end, ""))
            else:
                replacements.append((start, end, url))

    # Apply replacements in reverse order
    if replacements:
        replacements.sort(key=lambda x: x[0], reverse=True)
        for start, end, replacement in replacements:
            output = output[:start] + replacement + output[end:]

    return links, output


def get_md_header_contents(
    md_text: str,
    headers_to_split_on: List[Tuple[str, str]] = [],
    ignore_links: bool = True,
    base_url: Optional[str] = None
) -> List[HeaderDocument]:
    """
    Parses a Markdown string and splits it into a list of HeaderDocument instances,
    each containing the header text, parent header, header level, content, and links.
    Relationships are derived using HeaderDocument.from_list.

    Args:
        md_text: The Markdown (or HTML) text to parse and split.
        headers_to_split_on: List of (prefix, tag) pairs to identify headers.
        ignore_links: If True, replaces links with text content in links output; if False, preserves links in links output.
        base_url: Base URL to resolve relative links.

    Returns:
        List of HeaderDocument instances with derived relationships.
    """
    title = ""
    if is_html(md_text):
        soup = BeautifulSoup(md_text, 'html.parser')
        title_tag = soup.find('title')
        title = title_tag.get_text().strip() if title_tag else ""
        md_text = convert_html_to_markdown(md_text)

    # Clean markdown
    md_text = clean_spaces(clean_newlines(
        clean_text(md_text), max_newlines=2, strip_lines=True))

    # Extract links with original ignore_links setting for link extraction
    all_links, _ = extract_markdown_links(
        md_text, base_url, ignore_links=ignore_links)

    # Extract cleaned text with links replaced by their labels for header/content/text
    _, cleaned_md_text = extract_markdown_links(
        md_text, base_url, ignore_links=True)

    # Default headers to split on
    if not headers_to_split_on:
        headers_to_split_on = [(r"^(#+)\s*(.*)$", "header")]

    result: List[HeaderDocumentDict] = []
    header_stack: List[Dict[str, Any]] = []
    lines = cleaned_md_text.splitlines()  # Use cleaned text for processing

    # Process headers from cleaned markdown
    headers = []
    for line_idx, line in enumerate(lines, 1):
        for pattern, _ in headers_to_split_on:
            match = re.match(pattern, line)
            if match:
                if pattern == r"^(#+)\s*(.*)$":
                    level = len(match.group(1))
                    # Remove residual markdown characters
                    header_text = re.sub(r'[!]+', '', match.group(2)).strip()
                    raw_line = f"{'#' * level} {header_text}".strip()
                else:
                    level = 1
                    header_text = re.sub(
                        r'[!]+', '', match.group(1)).strip() if match.groups() else line.strip()
                    raw_line = header_text.strip()
                if header_text or raw_line:
                    headers.append(
                        {"text": header_text, "level": level, "line": line_idx, "raw_line": raw_line})

    # Handle content before first header
    first_header_line = min((h["line"]
                            for h in headers), default=1) if headers else 1
    if title or first_header_line > 1:
        content_lines = [
            line for line in lines[:first_header_line-1] if line.strip()]
        content = "\n".join(content_lines)
        content = re.sub(r'\n{3,}', '\n\n', content) if content else ""
        content = clean_repeated_ngrams(content)
        content = content.strip()
        title = title.strip()
        if content or title:
            texts = [title] if title else content.splitlines()
            header_links = [
                link for link in all_links if link["line_idx"] < first_header_line - 1]
            result.append({
                "text": content,
                "metadata": {
                    "header": title,
                    "parent_header": "",
                    "header_level": 0,
                    "content": content,
                    "doc_index": 0,
                    "chunk_index": None,
                    "tokens": None,
                    "source_url": base_url,
                    "links": header_links,
                    "texts": texts,
                    "id": str(uuid.uuid4())
                }
            })

    # Build header sections
    for header_idx, header in enumerate(headers):
        level = header["level"]
        text = header["text"].strip()
        line_idx = header["line"]
        raw_line = header["raw_line"].strip()

        if not raw_line:
            continue

        # Determine parent header
        parent_header = ""
        while header_stack and header_stack[-1]["header_level"] >= level:
            header_stack.pop()
        if header_stack:
            parent_header = header_stack[-1]["header"]

        # Collect content
        content_lines = []
        next_header_line = min(
            (h["line"] for h in headers[header_idx + 1:]), default=len(lines) + 1)
        for i in range(line_idx - 1, next_header_line - 1):
            if i < len(lines):
                content_lines.append(lines[i])
        content = "\n".join(content_lines[1:])  # Exclude header line
        content = re.sub(r'\n{3,}', '\n\n', content) if content else ""
        content = clean_repeated_ngrams(content)
        full_text = raw_line + ("\n" + content if content else "")

        # Extract links for this section
        header_links = [link for link in all_links if line_idx -
                        1 <= link["line_idx"] < next_header_line - 1]

        texts = [raw_line] + content.splitlines() if content else [raw_line]
        result.append({
            "text": full_text,
            "metadata": {
                "header": raw_line,
                "parent_header": parent_header,
                "header_level": level,
                "content": content,
                "doc_index": header_idx + (1 if first_header_line > 1 else 0),
                "chunk_index": None,
                "tokens": None,
                "source_url": base_url,
                "links": header_links,
                "texts": texts,
                "id": str(uuid.uuid4())
            }
        })
        header_stack.append({"header": raw_line, "header_level": level})

    return HeaderDocument.from_list(result)


def get_md_header_docs(
    md_text: str,
    headers_to_split_on: List[Tuple[str, str]] = [],
    ignore_links: bool = False,
    metadata: Optional[HeaderMetadataDoc] = None
) -> List[HeaderDocument]:
    """
    Splits the given markdown text into header-based document chunks and returns a list of HeaderDocument objects.

    Args:
        md_text (str): The markdown text to split.
        headers_to_split_on (List[Tuple[str, str]], optional): List of (prefix, header) tuples to split on. Defaults to [].
        ignore_links (bool, optional): If True, ignores links in headers. Defaults to False.
        metadata (Optional[HeaderMetadataDoc], optional): Metadata to attach to each HeaderDocument. Defaults to None.

    Returns:
        List[HeaderDocument]: List of HeaderDocument objects, one for each header chunk.
    """
    # Extract base_url from metadata if available
    base_url = None
    if metadata and "source_url" in metadata:
        parsed_url = urlparse(metadata["source_url"])
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"

    headers = get_md_header_contents(
        md_text, headers_to_split_on, ignore_links, base_url)

    # Update each HeaderDocument with doc_index and metadata
    for i, header_doc in enumerate(headers):
        current_metadata = dict(header_doc.metadata)
        current_metadata["doc_index"] = i
        if metadata:
            current_metadata.update(metadata)
        header_doc.metadata = current_metadata

    return headers


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
