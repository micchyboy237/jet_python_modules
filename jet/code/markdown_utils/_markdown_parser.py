import json
from jet.code.html_utils import valid_html
from jet.code.markdown_utils._converters import add_list_table_header_placeholders
from jet.code.markdown_utils._utils import preprocess_custom_code_blocks
from jet.wordnet.sentence import is_list_sentence, is_ordered_list_marker, split_sentences_with_separators
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from jet.data.utils import generate_unique_id
from jet.transformers.object import make_serializable
from mrkdwn_analysis import MarkdownParser

from jet.code.markdown_types import MarkdownToken, HeaderDoc
from jet.code.markdown_utils import read_md_content, preprocess_markdown

from jet.logger import logger


# @timeout(3)
def base_parse_markdown(input: Union[str, Path], ignore_links: bool = False) -> List[MarkdownToken]:
    if valid_html(str(input)):
        input = add_list_table_header_placeholders(str(input))
    md_content = read_md_content(input, ignore_links=ignore_links)
    md_content = preprocess_markdown(md_content)
    md_content = preprocess_custom_code_blocks(md_content)
    parser = MarkdownParser(md_content)
    md_tokens: List[MarkdownToken] = make_serializable(parser.parse())

    # Split paragraph tokens by newlines and handle dict or list content
    split_tokens: List[MarkdownToken] = []
    for token in md_tokens:
        if token.get("type") in ("unordered_list", "ordered_list"):
            items = token.get("meta", {}).get("items", [])
            # First, pick out items that are headers (start with "#"), and rest are 'regular' list items
            header_items = []
            non_header_items = []
            for item in items:
                item_text = (item.get("text", "") or "").strip()
                if item_text.startswith("#"):
                    header_items.append(item)
                else:
                    non_header_items.append(item)
            # If we found header_items, add header token(s) before the list token
            for header_item in header_items:
                header_text = header_item.get("text", "").strip()
                # Determine header level by number of leading '#'
                header_level = len(header_text) - len(header_text.lstrip("#"))
                header_content = header_text.lstrip("#").strip()
                # Only add as header if there is some content
                if header_content:
                    split_tokens.append({
                        'type': 'header',
                        'content': header_content,
                        'level': header_level,
                        'meta': {},
                        'line': token.get('line', 1)
                    })
            # Filter out items without any alphanumeric character
            filtered_items = [item for item in non_header_items if any(c.isalnum() for c in item.get("text", ""))]
            lines = []
            for item in filtered_items:
                text = item.get("text", "")
                lines.append(f"- {text}")
            merged_content = "\n".join(lines)
            # Make a copy to prevent mutating input token
            list_token = dict(token)
            list_token["content"] = merged_content
            # Also overwrite the filtered items in the meta
            list_token["meta"] = dict(list_token.get("meta", {}))
            list_token["meta"]["items"] = filtered_items
            split_tokens.append(list_token)
        elif isinstance(token.get('content'), (dict, list)):
            # Handle dict or list content by setting type to 'json' and converting to string
            split_tokens.append({
                'type': 'json',
                'content': f"```json{json.dumps(token['content'])}```",
                'level': None,
                'meta': token.get('meta', {}),
                'line': token.get('line', 1)
            })
        elif token.get('type', 'paragraph') == 'paragraph' and '\n' in token.get('content', ''):
            # Split paragraph tokens by newlines
            lines = token['content'].split('\n')
            for i, line in enumerate(lines):
                if line.strip():  # Only include non-empty lines
                    split_tokens.append({
                        'type': 'paragraph',
                        'content': line,
                        'level': None,
                        'meta': token.get('meta', {}),
                        'line': token.get('line', 1) + i
                    })
        else:
            split_tokens.append(token)

    return remove_header_placeholders(split_tokens)


def parse_markdown(input: Union[str, Path], merge_contents: bool = True, merge_headers: bool = True, ignore_links: bool = False) -> List[MarkdownToken]:
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
        def prepend_hashtags_to_headers(markdown_tokens: List[MarkdownToken]) -> List[MarkdownToken]:
            """Prepend hashtags to header tokens based on their level."""
            for token in markdown_tokens:
                if token['type'] == 'header' and token['level']:
                    hashtags = '#' * token['level']
                    if not token['content'].startswith(hashtags):
                        token['content'] = f"{hashtags} {token['content']}"
            return markdown_tokens

        tokens = base_parse_markdown(input, ignore_links=ignore_links)
        tokens = prepend_missing_headers_by_type(tokens)
        tokens = remove_leading_non_headers(tokens)
        if merge_contents:
            tokens = merge_tokens(tokens)
        tokens = prepend_hashtags_to_headers(tokens)
        if merge_contents:
            tokens = merge_tokens(tokens)
        if merge_headers:
            tokens = merge_headers_with_content(tokens)
        parsed_tokens = [
            {
                "type": token['type'],
                "content": derive_text(token),
                "level": token.get("level"),
                "meta": token.get("meta"),
                "line": token.get('line')
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


def derive_by_header_hierarchy(md_content: str, ignore_links: bool = False, valid_sentences_only: bool = False) -> List[HeaderDoc]:
    tokens = parse_markdown(
        md_content, merge_headers=False, merge_contents=False, ignore_links=ignore_links)
    sections: List[HeaderDoc] = []
    current_section: Optional[HeaderDoc] = None
    header_stack: List[tuple[str, int, int]] = []
    current_tokens = []
    section_index = 0

    for token in tokens:
        token_type = token.get("type")
        token_content = token.get("content", "")
        token_level = token.get("level", None)

        if valid_sentences_only:
            valid_sentences = split_sentences_with_separators(token_content, valid_only=valid_sentences_only)
            token_content = "".join(valid_sentences)

        if token_type == "header":
            if current_section:
                current_section["content"] = "\n".join(
                    current_section["content"])
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
                "source": '',
                "tokens": [],
            }
            header_stack.append(
                (current_section["header"], token_level, section_index))
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
                    "source": '',
                    "tokens": [],
                }
            current_section["content"].extend(token_content.splitlines())
            current_tokens.append(token)

    if current_section:
        current_section["content"] = "\n".join(current_section["content"])
        current_section["tokens"] = current_tokens
        sections.append(current_section)

    # Filter to exclude sections with empty headers or content
    sections = [section for section in sections if section.get(
        "header", "").strip() and section.get("content", "").strip()]

    for idx, section in enumerate(sections):
        section["doc_index"] = idx

    return sections


def derive_text(token: MarkdownToken) -> str:
    """
    Derives the Markdown text representation for a given token based on its type.
    Applies specific content transformations for code and unordered list tokens.
    """
    result = ""

    if token['type'] == 'header' and token['level'] is not None:
        result = f"{token['content'].strip()}" if token['content'] else ""

    elif token['type'] in ['unordered_list', 'ordered_list']:
        if not token['meta'] or 'items' not in token['meta']:
            result = ""
        else:
            items = token['meta']['items']
            prefix = '*' if token['type'] == 'unordered_list' else lambda i: f"{i+1}."
            lines = []
            for i, item in enumerate(items):
                checkbox = '[x]' if item.get('checked', False) else '[ ]' if item.get(
                    'task_item', False) else ''
                prefix_str = prefix(
                    i) if token['type'] == 'ordered_list' else prefix
                item_text = item['text'].strip()
                # Check if item text already starts with a list marker
                if token['type'] == 'ordered_list' and not is_ordered_list_marker(item_text.split()[0]):
                    # Prepend prefix for ordered list if no ordered marker exists
                    line = f"{prefix_str} {checkbox}{' ' if checkbox else ''}{item_text}".strip(
                    )
                elif token['type'] == 'unordered_list' and not is_list_sentence(item_text):
                    # Prepend prefix for unordered list if no list marker exists
                    line = f"{prefix_str} {checkbox}{' ' if checkbox else ''}{item_text}".strip(
                    )
                else:
                    # Use item text as is, ensuring checkbox is included if present
                    line = f"{checkbox}{' ' if checkbox else ''}{item_text}".strip()
                if token['type'] == 'unordered_list' and prefix_str == '*' and not is_list_sentence(item_text):
                    # Replace asterisk with dash for unordered lists if no marker exists
                    line = line.replace('* ', '- ')
                lines.append(line)
            result = '\n'.join(lines)

    elif token['type'] == 'table':
        if not token['meta'] or 'header' not in token['meta'] or 'rows' not in token['meta']:
            result = ""
        else:
            header = token['meta']['header']
            rows = token['meta']['rows']
            # Calculate widths based on header and rows, considering only header's column count
            widths = [max(len(str(cell)) for row in [
                          header] + rows for cell in row[i:i+1] or [""]) for i in range(len(header))]
            lines = ['| ' + ' | '.join(cell.ljust(widths[i])
                                       for i, cell in enumerate(header)) + ' |']
            lines.append(
                '| ' + ' | '.join('-' * width for width in widths) + ' |')
            for row in rows:
                # Pad or truncate row to match header length
                adjusted_row = (row + [""] * len(header))[:len(header)]
                lines.append(
                    '| ' + ' | '.join(str(cell).ljust(widths[i]) for i, cell in enumerate(adjusted_row)) + ' |')
            result = '\n'.join(lines)

    elif token['type'] == 'code':
        content = token['content']
        # Remove code block delimiters and strip whitespace
        result = re.sub(r'^```[\w]*\n|\n```$', '', content).strip()

    else:  # paragraph, blockquote, html_block
        result = token.get('content', '')

    return result.strip()


def merge_tokens(tokens: List[MarkdownToken]) -> List[MarkdownToken]:
    result: List[MarkdownToken] = []
    paragraph_buffer: List[str] = []
    list_buffer: List[Dict[str, Any]] = []
    list_content_buffer: List[str] = []
    current_line: int = 1

    for token in tokens:
        if token['type'] == 'paragraph':
            if not paragraph_buffer:  # First paragraph in sequence
                current_line = token.get('line', 1)
            paragraph_buffer.append(token['content'].strip())
        elif token['type'] == 'unordered_list':
            if not list_buffer:  # First unordered list in sequence
                current_line = token.get('line', 1)
            items = token.get('meta', {}).get('items', [])
            list_buffer.extend(items)
            # Generate content directly from items to ensure correct format
            list_content_buffer.append(
                '\n'.join(f"- {item['text']}" for item in items))
        else:
            if paragraph_buffer:
                merged_content = '\n'.join(paragraph_buffer)
                result.append({
                    'type': 'paragraph',
                    'content': merged_content,
                    'level': None,
                    'meta': {},
                    'line': current_line
                })
                paragraph_buffer = []
                current_line = 1
            if list_buffer:
                merged_content = '\n'.join(list_content_buffer)
                result.append({
                    'type': 'unordered_list',
                    'content': merged_content,
                    'level': None,
                    'meta': {'items': list_buffer},
                    'line': current_line
                })
                list_buffer = []
                list_content_buffer = []
                current_line = 1
            result.append(token)
            current_line = 1  # Reset for next sequence

    # Handle remaining buffers
    if paragraph_buffer:
        merged_content = '\n'.join(paragraph_buffer)
        result.append({
            'type': 'paragraph',
            'content': merged_content,
            'level': None,
            'meta': {},
            'line': current_line
        })
    if list_buffer:
        merged_content = '\n'.join(list_content_buffer)
        result.append({
            'type': 'unordered_list',
            'content': merged_content,
            'level': None,
            'meta': {'items': list_buffer},
            'line': current_line
        })

    return result


def remove_header_placeholders(markdown_tokens: List[MarkdownToken]) -> List[MarkdownToken]:
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
            "is_placeholder": token['type'] == 'header' and token.get('content', '') and 'placeholder' in token['content']
        }
        for i, token in enumerate(markdown_tokens)
        if token['type'] == 'header'
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
            last_header_non_placeholder_index + 1]["index"]
    else:
        # If all headers are placeholders or no placeholders follow, keep all tokens up to the last one
        consecutive_placeholder_header_first_index = len(markdown_tokens)

    # Process tokens up to the first consecutive placeholder header index
    filtered_tokens = []
    for index, token in enumerate(markdown_tokens[:consecutive_placeholder_header_first_index]):
        content = token.get('content', '')
        is_placeholder = token['type'] == 'header' and content and 'placeholder' in content

        if is_placeholder:
            # Check if content has a value (non-empty after stripping)
            if content.strip():
                # Split content by lines and remove the first line
                lines = content.split('\n')
                new_content = '\n'.join(lines[1:]) if len(lines) > 1 else ''
                # Only append if the remaining content is non-empty
                if new_content.strip():
                    filtered_tokens.append({
                        'type': 'header',
                        'content': new_content,
                        'level': token.get('level', 1),
                        'meta': token.get('meta', {}),
                        'line': token.get('line', 0)
                    })
            continue

        filtered_tokens.append(token)

    return filtered_tokens


def remove_leading_non_headers(markdown_tokens: List[MarkdownToken]) -> List[MarkdownToken]:
    """Remove all tokens at the start until the first header token is encountered, unless no headers exist."""
    for i, token in enumerate(markdown_tokens):
        if token['type'] == 'header':
            filtered_tokens = markdown_tokens[i:]
            return filtered_tokens
    return markdown_tokens.copy()


def merge_headers_with_content(markdown_tokens: List[MarkdownToken]) -> List[MarkdownToken]:
    """Merge headers with their succeeding non-header tokens into a single header token with content joined by newlines."""
    merged_tokens: List[MarkdownToken] = []
    current_header: MarkdownToken | None = None
    content_buffer: List[str] = []

    for token in markdown_tokens:
        if token['type'] == 'header':
            if current_header:
                # Finalize previous header
                merged_content = '\n'.join(content_buffer)
                merged_tokens.append({
                    'type': 'header',
                    'content': merged_content,
                    'level': current_header['level'],
                    'meta': current_header.get('meta', {}),
                    'line': current_header['line']
                })
                content_buffer = []
            current_header = token
            content_buffer.append(token['content'])
        else:
            if current_header:
                content = token.get('content')
                if content is None and token['type'] == 'unordered_list':
                    items = token.get('meta', {}).get('items', [])
                    content = '\n'.join(f"- {item['text']}" for item in items)
                # Add debug log for ordered_list
                if content is None and token['type'] == 'ordered_list':
                    items = token.get('meta', {}).get('items', [])
                    content = '\n'.join(
                        f"{i+1}. {item['text']}" for i, item in enumerate(items))
                if content:
                    content_buffer.append(content)

    # Finalize the last header
    if current_header and content_buffer:
        merged_content = '\n'.join(content_buffer)
        merged_tokens.append({
            'type': 'header',
            'content': merged_content,
            'level': current_header['level'],
            'meta': current_header.get('meta', {}),
            'line': current_header['line']
        })

    return merged_tokens


def prepend_missing_headers_by_type(tokens: List[MarkdownToken]) -> List[MarkdownToken]:
    result: List[MarkdownToken] = []
    last_header = None
    last_header_level = 2  # Default level, updated when a header is encountered
    pending_paragraphs: List[MarkdownToken] = []
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
                        "content": last_header.lstrip('# ').strip(),
                        "line": last_result["line"],
                        "type": "header",
                        # Use the level from the last header
                        "level": last_result["level"],
                        "meta": {}
                    }
                else:
                    # Append new header if the last header doesn't match
                    new_header = {
                        "content": last_header.lstrip('# ').strip(),
                        "line": current_line,
                        "type": "header",
                        "level": last_header_level,  # Use the stored header level
                        "meta": {}
                    }
                    result.append(new_header)
                    current_line += 1
            elif last_header:
                # Append new header if no header exists at the end
                new_header = {
                    "content": last_header.lstrip('# ').strip(),
                    "line": current_line,
                    "type": "header",
                    "level": last_header_level,  # Use the stored header level
                    "meta": {}
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
