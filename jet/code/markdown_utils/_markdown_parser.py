import json
import re
from pathlib import Path
from typing import Any

from jet.code.html_utils import valid_html
from jet.code.markdown_types import HeaderDoc, MarkdownToken
from jet.code.markdown_utils import preprocess_markdown, read_md_content
from jet.code.markdown_utils._converters import add_list_table_header_placeholders
from jet.code.markdown_utils._utils import preprocess_custom_code_blocks
from jet.data.utils import generate_unique_id
from jet.logger import logger
from jet.scrapers.utils import scrape_metadata, scrape_title
from jet.transformers.object import make_serializable
from jet.wordnet.sentence import (
    is_list_sentence,
    is_ordered_list_marker,
    split_sentences_with_separators,
)
from mrkdwn_analysis import MarkdownParser


# @timeout(3)
def flatten_nested_content(
    content: Any,
    depth: int = 0,
    prefix: str = "",
    is_ordered: bool = False,
    item_index: int = 0,
) -> str:
    """
    Recursively flatten potentially nested markdown content (str, list, dict) into
    a single markdown-ish string with proper indentation/prefixes.
    Used for list items and table cells that may contain sub-lists, paragraphs, etc.
    """
    if isinstance(content, str):
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        if not lines:
            return ""
        indent = "  " * depth
        return "\n".join(f"{indent}{prefix}{line}" for line in lines)

    elif isinstance(content, list):
        parts = []
        for i, item in enumerate(content):
            sub_prefix = f"{i + 1}. " if is_ordered else "- "
            sub = flatten_nested_content(
                item,
                depth=depth + 1,
                prefix=sub_prefix,
                is_ordered=is_ordered,
                item_index=i,
            )
            if sub.strip():
                parts.append(sub)
        return "\n".join(parts)

    elif isinstance(content, dict):
        # Handle common nested token-like dicts
        if "text" in content:
            return flatten_nested_content(
                content["text"],
                depth=depth,
                prefix=prefix,
                is_ordered=is_ordered,
            )
        elif "items" in content:
            return flatten_nested_content(
                content["items"],
                depth=depth,
                prefix=prefix,
                is_ordered=content.get("ordered", False),
            )
        elif "rows" in content or "header" in content:
            return "(nested table — not yet deeply flattened)"
        else:
            try:
                return json.dumps(content, ensure_ascii=False, indent=2)
            except Exception:
                return str(content)

    return str(content)


def base_parse_markdown(
    input: str | Path, ignore_links: bool = False
) -> list[MarkdownToken]:
    if valid_html(str(input)):
        input = add_list_table_header_placeholders(str(input))
    md_content = read_md_content(input, ignore_links=ignore_links)
    md_content = preprocess_markdown(md_content)
    md_content = preprocess_custom_code_blocks(md_content)
    parser = MarkdownParser(md_content)
    md_tokens: list[MarkdownToken] = make_serializable(parser.parse())

    split_tokens: list[MarkdownToken] = []
    for token in md_tokens:
        if token.get("type") in ("unordered_list", "ordered_list"):
            items = token.get("meta", {}).get("items", [])
            is_ordered = token["type"] == "ordered_list"

            # Separate header items (start with '#') vs regular list items
            header_items = []
            non_header_items = []
            for item in items:
                item_text = (item.get("text", "") or "").strip()
                if item_text.startswith("#"):
                    header_items.append(item)
                else:
                    non_header_items.append(item)
            # Add header tokens if found
            for header_item in header_items:
                header_text = header_item.get("text", "").strip()
                header_level = len(header_text) - len(header_text.lstrip("#"))
                header_content = header_text.lstrip("#").strip()
                if header_content:
                    split_tokens.append(
                        {
                            "type": "header",
                            "content": header_content,
                            "level": header_level,
                            "meta": {},
                            "line": token.get("line", 1),
                        }
                    )

            # Filter out items without any alphanumeric in "text"
            filtered_items = [
                item
                for item in non_header_items
                if any(c.isalnum() for c in item.get("text", ""))
            ]

            flattened_items = []
            md_lines = []
            for i, item in enumerate(filtered_items):
                flat_text = flatten_nested_content(
                    item.get("text", ""),
                    depth=0,
                    prefix="",
                    is_ordered=is_ordered,
                    item_index=i,
                ).strip()
                if not flat_text:
                    continue

                checkbox = ""
                if item.get("task_item"):
                    checkbox = "[x] " if item.get("checked") else "[ ] "

                prefix = f"{i + 1}. " if is_ordered else "- "
                line = f"{prefix}{checkbox}{flat_text}"
                md_lines.append(line)
                flattened_items.append(
                    {
                        **item,
                        "text": flat_text,  # flattened
                    }
                )

            if not md_lines:
                continue

            merged_content = "\n".join(md_lines)
            list_token = dict(token)
            list_token["content"] = merged_content
            list_token["meta"] = dict(list_token.get("meta", {}))
            list_token["meta"]["items"] = flattened_items
            split_tokens.append(list_token)

        elif isinstance(token.get("content"), (dict, list)):
            split_tokens.append(
                {
                    "type": "json",
                    "content": f"```json{json.dumps(token['content'])}```",
                    "level": None,
                    "meta": token.get("meta", {}),
                    "line": token.get("line", 1),
                }
            )

        elif token.get("type", "paragraph") == "paragraph" and "\n" in token.get(
            "content", ""
        ):
            lines = token["content"].split("\n")
            for i, line in enumerate(lines):
                if line.strip():
                    split_tokens.append(
                        {
                            "type": "paragraph",
                            "content": line,
                            "level": None,
                            "meta": token.get("meta", {}),
                            "line": token.get("line", 1) + i,
                        }
                    )

        elif token.get("type") == "table":
            table_meta = token.get("meta", {})
            if not isinstance(table_meta, dict):
                split_tokens.append(token)
                continue

            header = table_meta.get("header", [])
            rows = table_meta.get("rows", [])

            # Flatten table header and rows recursively
            flat_header = [
                flatten_nested_content(cell, depth=0).strip().replace("\n", " <br> ")
                for cell in header
            ]

            flat_rows = []
            for row in rows:
                flat_row = [
                    flatten_nested_content(cell, depth=0)
                    .strip()
                    .replace("\n", " <br> ")
                    for cell in row
                ]
                flat_rows.append(flat_row)

            # Build markdown table string
            if flat_header:
                header_row = "| " + " | ".join(flat_header) + " |"
                separator = "| " + " | ".join("---" for _ in flat_header) + " |"
                body_rows = "\n".join(
                    "| " + " | ".join(row) + " |" for row in flat_rows
                )
                content = "\n".join([header_row, separator, body_rows])
            else:
                content = "\n".join("| " + " | ".join(row) + " |" for row in flat_rows)

            table_token = dict(token)
            table_token["content"] = content
            table_token["meta"] = {
                "header": flat_header,
                "rows": flat_rows,
            }
            split_tokens.append(table_token)
        else:
            split_tokens.append(token)

    tokens = remove_header_placeholders(split_tokens)

    # Insert document title if valid html
    input_str = str(input) if isinstance(input, Path) else input
    if valid_html(input_str):
        title_text = scrape_title(input_str)

        # Fallback if nothing found
        if not title_text:
            title_text = "Untitled Document"

        title_token: MarkdownToken = {
            "type": "head",
            "content": title_text,
            "level": None,
            "meta": scrape_metadata(input_str),
            "line": 0,
        }

        # Insert title token at the beginning
        tokens = [title_token] + tokens

    return tokens


def parse_markdown(
    input: str | Path,
    merge_contents: bool = True,
    merge_headers: bool = True,
    ignore_links: bool = False,
) -> list[MarkdownToken]:
    """
    Parse markdown content into a list of structured tokens using MarkdownParser.
    Args:
        input: Either a string containing markdown content or a Path to a markdown file.
        merge_contents: If True, merge consecutive paragraph and unordered list tokens into single tokens. Defaults to True.
        merge_headers: If True, merge headers with their succeeding non-header tokens into single header tokens. Defaults to False.
        ignore_links: If True, remove or ignore links during HTML to Markdown conversion. Defaults to False.
    Returns:
        A list of dictionaries representing parsed markdown tokens with their type, content, and metadata.
    Raises:
        OSError: If the input file does not exist.
        TimeoutError: If parsing takes too long.
    """
    try:

        def prepend_hashtags_to_headers(
            markdown_tokens: list[MarkdownToken],
        ) -> list[MarkdownToken]:
            """Prepend hashtags to header tokens based on their level."""
            for token in markdown_tokens:
                if token["type"] == "header" and token["level"]:
                    hashtags = "#" * token["level"]
                    if not token["content"].startswith(hashtags):
                        token["content"] = f"{hashtags} {token['content']}"
            return markdown_tokens

        tokens = base_parse_markdown(input, ignore_links=ignore_links)
        tokens = prepend_missing_headers_by_type(tokens)
        # tokens = remove_leading_non_headers(tokens)
        if merge_contents:
            tokens = merge_tokens(tokens)
        tokens = prepend_hashtags_to_headers(tokens)
        if merge_contents:
            tokens = merge_tokens(tokens)
        if merge_headers:
            tokens = merge_headers_with_content(tokens)
        parsed_tokens = [
            {
                "type": token["type"],
                "content": derive_text(token),
                "level": token.get("level"),
                "meta": token.get("meta"),
                "line": token.get("line"),
            }
            for token in tokens
        ]
        return parsed_tokens
    except TimeoutError as e:
        logger.error(f"Parsing timed out: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error parsing markdown: {str(e)}")
        raise


def derive_by_header_hierarchy(
    md_content: str, ignore_links: bool = False, valid_sentences_only: bool = False
) -> list[HeaderDoc]:
    tokens = parse_markdown(
        md_content, merge_headers=False, merge_contents=False, ignore_links=ignore_links
    )
    sections: list[HeaderDoc] = []
    current_section: HeaderDoc | None = None
    header_stack: list[tuple[str, int, int]] = []
    current_tokens = []
    section_index = 0

    for token in tokens:
        token_type = token.get("type")
        token_content = token.get("content", "")
        token_level = token.get("level", None)

        if valid_sentences_only:
            valid_sentences = split_sentences_with_separators(
                token_content, valid_only=valid_sentences_only
            )
            token_content = "".join(valid_sentences)

        if token_type == "header":
            if current_section:
                current_section["content"] = "\n".join(current_section["content"])
                current_section["tokens"] = current_tokens
                sections.append(current_section)
                section_index += 1

            current_tokens = [token]

            # Build parent_headers list
            parent_headers = []
            parent_header = None
            parent_level = None
            while header_stack and header_stack[-1][1] >= token_level:
                header_stack.pop()
            if header_stack:
                parent_header = header_stack[-1][0]
                parent_level = header_stack[-1][1]
                # Collect all parent headers up to root
                parent_headers = [h[0] for h in header_stack]

            current_section: HeaderDoc = {
                "id": generate_unique_id(),
                "doc_index": section_index,
                "header": token_content.splitlines()[0] if token_content else "",
                "content": [],
                "level": token_level,
                "parent_headers": parent_headers,
                "parent_header": parent_header,
                "parent_level": parent_level,
                "source": "",
                "tokens": [],
            }
            header_stack.append((current_section["header"], token_level, section_index))
        else:
            if current_section is None:
                current_section: HeaderDoc = {
                    "id": generate_unique_id(),
                    "doc_index": section_index,
                    "header": "",
                    "content": [],
                    "level": 0,
                    "parent_headers": [],
                    "parent_header": None,
                    "parent_level": None,
                    "source": "",
                    "tokens": [],
                }
            current_section["content"].extend(token_content.splitlines())
            current_tokens.append(token)

    if current_section:
        current_section["content"] = "\n".join(current_section["content"])
        current_section["tokens"] = current_tokens
        sections.append(current_section)

    # Filter to exclude sections with empty headers or content
    sections = [
        section
        for section in sections
        if section.get("header", "").strip() and section.get("content", "").strip()
    ]

    for idx, section in enumerate(sections):
        section["doc_index"] = idx

    return sections


def derive_text(token: MarkdownToken) -> str:
    """
    Derives the Markdown text representation for a given token based on its type.
    Applies specific content transformations for code and unordered list tokens.
    """
    result = ""

    if token["type"] == "header" and token["level"] is not None:
        result = f"{token['content'].strip()}" if token["content"] else ""

    elif token["type"] in ["unordered_list", "ordered_list"]:
        if not token["meta"] or "items" not in token["meta"]:
            result = ""
        else:
            items = token["meta"]["items"]
            prefix = "*" if token["type"] == "unordered_list" else lambda i: f"{i + 1}."
            lines = []
            for i, item in enumerate(items):
                checkbox = (
                    "[x]"
                    if item.get("checked", False)
                    else "[ ]"
                    if item.get("task_item", False)
                    else ""
                )
                prefix_str = prefix(i) if token["type"] == "ordered_list" else prefix
                item_text = item["text"].strip()
                # Check if item text already starts with a list marker
                if token["type"] == "ordered_list" and not is_ordered_list_marker(
                    item_text.split()[0]
                ):
                    # Prepend prefix for ordered list if no ordered marker exists
                    line = f"{prefix_str} {checkbox}{' ' if checkbox else ''}{item_text}".strip()
                elif token["type"] == "unordered_list" and not is_list_sentence(
                    item_text
                ):
                    # Prepend prefix for unordered list if no list marker exists
                    line = f"{prefix_str} {checkbox}{' ' if checkbox else ''}{item_text}".strip()
                else:
                    # Use item text as is, ensuring checkbox is included if present
                    line = f"{checkbox}{' ' if checkbox else ''}{item_text}".strip()
                if (
                    token["type"] == "unordered_list"
                    and prefix_str == "*"
                    and not is_list_sentence(item_text)
                ):
                    # Replace asterisk with dash for unordered lists if no marker exists
                    line = line.replace("* ", "- ")
                lines.append(line)
            result = "\n".join(lines)

    elif token["type"] == "table":
        if (
            not token["meta"]
            or "header" not in token["meta"]
            or "rows" not in token["meta"]
        ):
            result = ""
        else:
            header = token["meta"]["header"]
            rows = token["meta"]["rows"]
            # Calculate widths based on header and rows, considering only header's column count
            widths = [
                max(
                    len(str(cell))
                    for row in [header] + rows
                    for cell in row[i : i + 1] or [""]
                )
                for i in range(len(header))
            ]
            lines = [
                "| "
                + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(header))
                + " |"
            ]
            lines.append("| " + " | ".join("-" * width for width in widths) + " |")
            for row in rows:
                # Pad or truncate row to match header length
                adjusted_row = (row + [""] * len(header))[: len(header)]
                lines.append(
                    "| "
                    + " | ".join(
                        str(cell).ljust(widths[i])
                        for i, cell in enumerate(adjusted_row)
                    )
                    + " |"
                )
            result = "\n".join(lines)

    elif token["type"] == "code":
        content = token["content"]
        # Remove code block delimiters and strip whitespace
        result = re.sub(r"^```[\w]*\n|\n```$", "", content).strip()

    else:  # paragraph, blockquote, html_block
        result = token.get("content", "")

    return result.strip()


def merge_tokens(tokens: list[MarkdownToken]) -> list[MarkdownToken]:
    result: list[MarkdownToken] = []

    # Buffers for different mergeable block types
    paragraph_buffer: list[str] = []
    ul_buffer_items: list[dict[str, Any]] = []  # for unordered_list
    ol_buffer_items: list[dict[str, Any]] = []  # for ordered_list

    current_line: int = 1
    current_list_type: str | None = None  # "ul" or "ol" when buffering list

    def flush_paragraph():
        nonlocal current_line
        if paragraph_buffer:
            merged = "\n".join(paragraph_buffer)
            result.append(
                {
                    "type": "paragraph",
                    "content": merged,
                    "level": None,
                    "meta": {},
                    "line": current_line,
                }
            )
            paragraph_buffer.clear()
            current_line = 1

    def flush_list():
        nonlocal current_line, current_list_type
        if not ul_buffer_items and not ol_buffer_items:
            return

        if ul_buffer_items:
            content = "\n".join(f"- {item['text']}" for item in ul_buffer_items)
            result.append(
                {
                    "type": "unordered_list",
                    "content": content,
                    "level": None,
                    "meta": {"items": ul_buffer_items[:]},
                    "line": current_line,
                }
            )
            ul_buffer_items.clear()

        if ol_buffer_items:
            content = "\n".join(
                f"{i + 1}. {item['text']}" for i, item in enumerate(ol_buffer_items)
            )
            result.append(
                {
                    "type": "ordered_list",
                    "content": content,
                    "level": None,
                    "meta": {"items": ol_buffer_items[:]},
                    "line": current_line,
                }
            )
            ol_buffer_items.clear()

        current_list_type = None
        current_line = 1

    for token in tokens:
        ttype = token.get("type")

        # ─────────────────────────────────────
        # Paragraph handling
        # ─────────────────────────────────────
        if ttype == "paragraph":
            # If we were buffering a list → flush it first
            flush_list()

            if not paragraph_buffer:
                current_line = token.get("line", 1)
            paragraph_buffer.append(token["content"].strip())

        # ─────────────────────────────────────
        # Unordered list handling
        # ─────────────────────────────────────
        elif ttype == "unordered_list":
            flush_paragraph()  # different block type → flush paragraph

            items = token.get("meta", {}).get("items", [])

            if current_list_type == "ol":
                flush_list()

            if not ul_buffer_items and not ol_buffer_items:
                current_line = token.get("line", 1)
                current_list_type = "ul"

            ul_buffer_items.extend(items)

        # ─────────────────────────────────────
        # Ordered list handling (new)
        # ─────────────────────────────────────
        elif ttype == "ordered_list":
            flush_paragraph()

            items = token.get("meta", {}).get("items", [])

            if current_list_type == "ul":
                flush_list()

            if not ul_buffer_items and not ol_buffer_items:
                current_line = token.get("line", 1)
                current_list_type = "ol"

            ol_buffer_items.extend(items)

        # ─────────────────────────────────────
        # Any other block type
        # ─────────────────────────────────────
        else:
            flush_paragraph()
            flush_list()
            result.append(token)
            current_line = 1

    # Don't forget remaining buffered content
    flush_paragraph()
    flush_list()

    return result


def remove_header_placeholders(
    markdown_tokens: list[MarkdownToken],
) -> list[MarkdownToken]:
    """Remove placeholder header tokens and their succeeding non-header tokens at the end of the list by removing all tokens
    with indices >= the first consecutive placeholder header index. For placeholder headers in the middle, retain only if
    content has a value, then remove the first line of content."""
    # If the input list is empty, return empty list
    if not markdown_tokens:
        return []

    # Find all header token indices and their placeholder status
    all_header_token_indices = [
        {
            "index": i,
            "is_placeholder": token["type"] == "header"
            and token.get("content", "")
            and "placeholder" in token["content"],
        }
        for i, token in enumerate(markdown_tokens)
        if token["type"] == "header"
    ]

    # If no headers exist, return all tokens (no placeholder headers to process)
    if not all_header_token_indices:
        return markdown_tokens.copy()

    # Find the last non-placeholder header index
    last_header_non_placeholder_index = -1
    for header_info in reversed(all_header_token_indices):
        if not header_info["is_placeholder"]:
            last_header_non_placeholder_index = header_info["index"]
            break

    # Determine the first index of consecutive placeholder headers at the end
    if last_header_non_placeholder_index + 1 < len(all_header_token_indices):
        consecutive_placeholder_header_first_index = all_header_token_indices[
            last_header_non_placeholder_index + 1
        ]["index"]
    else:
        # If all headers are placeholders or no placeholders follow, keep all tokens up to the last one
        consecutive_placeholder_header_first_index = len(markdown_tokens)

    # Process tokens up to the first consecutive placeholder header index
    filtered_tokens = []
    for index, token in enumerate(
        markdown_tokens[:consecutive_placeholder_header_first_index]
    ):
        content = token.get("content", "")
        is_placeholder = (
            token["type"] == "header" and content and "placeholder" in content
        )

        if is_placeholder:
            # Check if content has a value (non-empty after stripping)
            if content.strip():
                # Split content by lines and remove the first line
                lines = content.split("\n")
                new_content = "\n".join(lines[1:]) if len(lines) > 1 else ""
                # Only append if the remaining content is non-empty
                if new_content.strip():
                    filtered_tokens.append(
                        {
                            "type": "header",
                            "content": new_content,
                            "level": token.get("level", 1),
                            "meta": token.get("meta", {}),
                            "line": token.get("line", 0),
                        }
                    )
            continue

        filtered_tokens.append(token)

    return filtered_tokens


def remove_leading_non_headers(
    markdown_tokens: list[MarkdownToken],
) -> list[MarkdownToken]:
    """Remove all tokens at the start until the first header token is encountered, unless no headers exist."""
    for i, token in enumerate(markdown_tokens):
        if token["type"] == "header":
            filtered_tokens = markdown_tokens[i:]
            return filtered_tokens
    return markdown_tokens.copy()


def merge_headers_with_content(
    markdown_tokens: list[MarkdownToken],
) -> list[MarkdownToken]:
    """Merge headers with their succeeding non-header tokens into a single header token with content joined by newlines."""
    merged_tokens: list[MarkdownToken] = []
    current_header: MarkdownToken | None = None
    content_buffer: list[str] = []

    for token in markdown_tokens:
        if token["type"] == "header":
            if current_header:
                # Finalize previous header
                merged_content = "\n".join(content_buffer)
                merged_tokens.append(
                    {
                        "type": "header",
                        "content": merged_content,
                        "level": current_header["level"],
                        "meta": current_header.get("meta", {}),
                        "line": current_header["line"],
                    }
                )
                content_buffer = []
            current_header = token
            content_buffer.append(token["content"])
        else:
            if current_header:
                content = token.get("content")
                if content is None and token["type"] == "unordered_list":
                    items = token.get("meta", {}).get("items", [])
                    content = "\n".join(f"- {item['text']}" for item in items)
                # Add debug log for ordered_list
                if content is None and token["type"] == "ordered_list":
                    items = token.get("meta", {}).get("items", [])
                    content = "\n".join(
                        f"{i + 1}. {item['text']}" for i, item in enumerate(items)
                    )
                if content:
                    content_buffer.append(content)

    # Finalize the last header
    if current_header and content_buffer:
        merged_content = "\n".join(content_buffer)
        merged_tokens.append(
            {
                "type": "header",
                "content": merged_content,
                "level": current_header["level"],
                "meta": current_header.get("meta", {}),
                "line": current_header["line"],
            }
        )

    return merged_tokens


def prepend_missing_headers_by_type(tokens: list[MarkdownToken]) -> list[MarkdownToken]:
    result: list[MarkdownToken] = []
    last_header = None
    last_header_level = 2  # Default level, updated when a header is encountered
    pending_paragraphs: list[MarkdownToken] = []
    current_line = 1  # Start at line 1

    for token in tokens:
        token_copy = token.copy()

        if token["type"] == "header":
            last_header = token["content"]
            last_header_level = token["level"]  # Update the header level
            # Append any pending paragraphs before the header
            for para in pending_paragraphs:
                para_copy = para.copy()
                para_copy["line"] = current_line
                result.append(para_copy)
                current_line += 1
            pending_paragraphs = []
            token_copy["line"] = current_line
            result.append(token_copy)
            current_line += 1

        elif token["type"] == "paragraph":
            pending_paragraphs.append(token_copy)

        # Non-header, non-paragraph token (e.g., unordered_list, blockquote, code)
        else:
            # Check if the last token in result is a header that needs replacement
            if last_header and result and result[-1]["type"] == "header":
                last_result = result[-1]
                if last_result["content"] == last_header:
                    # Replace the last header with one that does NOT include the token type
                    result[-1] = {
                        "content": last_header.lstrip("# ").strip(),
                        "line": last_result["line"],
                        "type": "header",
                        # Use the level from the last header
                        "level": last_result["level"],
                        "meta": {},
                    }
                else:
                    # Append new header if the last header doesn't match
                    new_header = {
                        "content": last_header.lstrip("# ").strip(),
                        "line": current_line,
                        "type": "header",
                        "level": last_header_level,  # Use the stored header level
                        "meta": {},
                    }
                    result.append(new_header)
                    current_line += 1
            elif last_header:
                # Append new header if no header exists at the end
                new_header = {
                    "content": last_header.lstrip("# ").strip(),
                    "line": current_line,
                    "type": "header",
                    "level": last_header_level,  # Use the stored header level
                    "meta": {},
                }
                result.append(new_header)
                current_line += 1

            # Append any pending paragraphs after the header but before the token
            for para in pending_paragraphs:
                para_copy = para.copy()
                para_copy["line"] = current_line
                result.append(para_copy)
                current_line += 1
            pending_paragraphs = []

            token_copy["line"] = current_line
            result.append(token_copy)
            current_line += 1

    # Append any remaining paragraphs
    for para in pending_paragraphs:
        para_copy = para.copy()
        para_copy["line"] = current_line
        result.append(para_copy)
        current_line += 1

    return result


__all__ = [
    "base_parse_markdown",
    "merge_tokens",
    "remove_header_placeholders",
    "remove_leading_non_headers",
    "merge_headers_with_content",
    "parse_markdown",
    "prepend_missing_headers_by_type",
    "derive_text",
    "derive_by_header_hierarchy",
]
