import os
import tempfile
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, TypedDict

import html2text
from markdownify import MarkdownConverter
from markitdown import MarkItDown
from jet.code.html_utils import preprocess_html, valid_html
from jet.code.markdown_types import MarkdownAnalysis, MarkdownToken, SummaryDict
from jet.decorators.timer import timeout
from jet.transformers.object import make_serializable
from jet.utils.text import fix_and_unidecode
from mrkdwn_analysis import MarkdownAnalyzer, MarkdownParser

from jet.logger import logger


def to_snake_case(s: str) -> str:
    """Convert a string to lowercase and underscore-separated, replacing spaces and normalizing underscores."""
    # Replace all whitespace with a single underscore
    s = re.sub(r'\s+', '_', s.strip())
    # Insert underscore before uppercase letters (except at start), then lowercase
    s = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s)
    # Convert to lowercase and normalize multiple underscores to one
    s = re.sub(r'_+', '_', s.lower())
    return s


def convert_dict_keys_to_snake_case(d: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively convert dictionary keys to snake_case."""
    if not isinstance(d, dict):
        return d
    return {
        to_snake_case(k): [convert_dict_keys_to_snake_case(item) for item in v]
        if isinstance(v, list) else convert_dict_keys_to_snake_case(v)
        if isinstance(v, dict) else v
        for k, v in d.items()
    }


def preprocess_markdown(md_content: str) -> str:
    """
    Preprocess markdown to normalize non-standard structures.

    Args:
        md_content: Raw markdown content.

    Returns:
        Normalized markdown content.
    """
    # Remove bold markdown syntax (e.g., **text** or __text__ to text)
    md_content = re.sub(r'\*\*(.*?)\*\*', r'\1', md_content)
    md_content = re.sub(r'__(.*?)__', r'\1', md_content)
    # Remove leading spaces from header lines
    md_content = re.sub(r'^\s*(#+)', r'\1', md_content, flags=re.MULTILINE)
    # Remove leading spaces from blockquote lines
    md_content = re.sub(r'^\s*(>+)', r'\1', md_content, flags=re.MULTILINE)
    # Ensure proper spacing around headers with links
    md_content = re.sub(r'^(#+)\s*\[([^\]]+)\]\(([^)]+)\)',
                        r'\1 [\2](\3)', md_content, flags=re.MULTILINE)

    # Remove bold markers but preserve headers
    md_content = re.sub(r'\*\*(.*?)\*\*', r'\1', md_content)
    md_content = re.sub(r'__(.*?)__', r'\1', md_content)
    # Fix incomplete regex for blockquotes
    md_content = re.sub(r'^\s*(>+)\s*', r'\1 ', md_content, flags=re.MULTILINE)
    # Fix task list regex to avoid affecting headers
    md_content = re.sub(r'^\s*([-*+])\s*\[([ xX])\]\s*(.*)',
                        r'\1 [\2] \3', md_content, flags=re.MULTILINE)

    # Process separator lines
    md_content = process_separator_lines(md_content)

    md_content = clean_markdown_text(md_content)
    return md_content


def process_separator_lines(md_content: str) -> str:
    """
    Process markdown content to handle lines with only '-' or '*', moving next line's text to the right
    with a single space, or removing the separator if no text follows.

    Args:
        md_content: Raw markdown content as a string.

    Returns:
        Processed markdown content with transformed separator lines.
    """
    lines = md_content.splitlines()
    result: List[str] = []
    i = 0

    while i < len(lines):
        current_line = lines[i].strip()
        if current_line in ("-", "*"):
            # Check if there's a next line
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line:
                    # Combine separator and next line text
                    result.append(f"{current_line} {next_line}")
                    i += 2  # Skip the next line
                else:
                    # No text in next line, skip the separator
                    i += 1
            else:
                # Separator is the last line, skip it
                i += 1
        else:
            # Preserve non-separator lines
            result.append(lines[i])
            i += 1

    # Join lines, preserving original line endings
    return "\n".join(result)


def clean_markdown_text(text: str) -> str:
    """
    Clean markdown text by removing unnecessary escape characters.

    Args:
        text: The markdown text to clean, or None.

    Returns:
        The cleaned text with unnecessary escapes removed, or None if input is None.
    """
    # Remove escaped periods (e.g., "\." -> ".")
    text = re.sub(r'\\([.])', r'\1', text)
    text = fix_and_unidecode(text)
    return text


def convert_html_to_markdownify(html_input: Union[str, Path], **options) -> str:
    """
    Convert HTML content to Markdown and return the string.
    """
    logger.info("Starting HTML to Markdown conversion")
    if isinstance(html_input, Path):
        with html_input.open('r', encoding='utf-8') as f:
            html_content = preprocess_html(f.read())
    else:
        html_content = preprocess_html(html_input)

    try:
        if isinstance(html_input, Path):
            with html_input.open('r', encoding='utf-8') as f:
                html_content = f.read()
        else:
            html_content = html_input

        # Convert to Markdown using markdownify
        markdown_content = MarkdownConverter(
            heading_style="ATX").convert(html_content)

        # Ensure consistent spacing after title, headers, and paragraphs
        markdown_content = re.sub(
            r'(?<!\n)\n(?=[#*-]|\w)', '\n\n', markdown_content.strip())
        markdown_content = re.sub(r'\n{3,}', '\n\n', markdown_content)

        return markdown_content

    except Exception as e:
        logger.error("Failed to convert HTML to Markdown: %s", e)
        raise


def convert_html_to_markitdown(html_input: Union[str, Path], **options) -> str:
    """
    Convert HTML content to Markdown using MarkItDown and return the string.

    Args:
        html_input: Either a string containing HTML content or a Path to an HTML file.
        **options: Additional options for MarkItDown conversion.

    Returns:
        Formatted Markdown content.
    """
    logger.info("Starting HTML to Markdown conversion with MarkItDown")
    try:
        md_converter = MarkItDown(enable_plugins=False, **options)

        if isinstance(html_input, Path):
            # Read HTML file content and preprocess
            with open(html_input, 'r', encoding='utf-8') as file:
                html_content = preprocess_html(file.read())
        else:
            # Preprocess HTML string
            html_content = preprocess_html(html_input)

        # Write preprocessed content to a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as temp_file:
            temp_file.write(html_content)
            temp_file_path = temp_file.name
        try:
            result = md_converter.convert(temp_file_path)
        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

        markdown_content = result.text_content
        return markdown_content

    except Exception as e:
        logger.error("Failed to convert HTML to Markdown: %s", e)
        raise


def add_list_table_placeholders(html: str) -> str:
    """
    Add <h6> placeholders after </ol>, </ul>, and </table> tags to prevent markdown parser issues.

    Args:
        html: Input HTML string to process.

    Returns:
        HTML string with <h6> placeholders added after specified tags.
    """
    return re.sub(r'</(ol|ul|table)>', r'</\1><h6>placeholder</h6>', html, flags=re.IGNORECASE)


def read_md_content(input) -> str:
    md_content = ""
    try:
        if Path(str(input)).is_file():
            with open(input, 'r', encoding='utf-8') as file:
                md_content = file.read()

            if str(input).endswith('.html'):
                md_content = convert_html_to_markdown(md_content)
    except OSError:
        md_content = str(input)

        if valid_html(md_content):
            md_content = convert_html_to_markdown(md_content)
    except Exception:
        raise
    return md_content


def parse_markdown(input: Union[str, Path], merge_contents: bool = False) -> List[MarkdownToken]:
    """
    Parse markdown content into a list of structured tokens using MarkdownParser.

    Args:
        input: Either a string containing markdown content or a Path to a markdown file.
        merge_contents: If True, merge consecutive paragraph and unordered list tokens into single tokens. Defaults to False.

    Returns:
        A list of dictionaries representing parsed markdown tokens with their type, content, and metadata.

    Raises:
        OSError: If the input file does not exist.
        TimeoutError: If parsing takes too long.
    """
    def merge_tokens(tokens: List[MarkdownToken]) -> List[MarkdownToken]:
        result: List[MarkdownToken] = []
        paragraph_buffer: List[str] = []
        list_buffer: List[Dict[str, Any]] = []
        list_content_buffer: List[str] = []
        current_line: Optional[int] = None

        for token in tokens:
            if token['type'] == 'paragraph':
                if current_line is None:
                    current_line = token['line']
                paragraph_buffer.append(token['content'].strip())
            elif token['type'] == 'unordered_list' and token.get('meta', {}).get('items'):
                if current_line is None:
                    current_line = token['line']
                list_buffer.extend(token['meta']['items'])
                list_content_buffer.append(derive_text(token))
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
                    current_line = None
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
                    current_line = None
                result.append(token)

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
    try:
        md_content = read_md_content(input)

        # Preprocess markdown
        md_content = preprocess_markdown(md_content)

        # @timeout(3)
        def parse_with_timeout(content: str) -> List[MarkdownToken]:
            parser = MarkdownParser(content)
            return make_serializable(parser.parse())

        def remove_list_table_placeholders(markdown_tokens: List[MarkdownToken]) -> List[MarkdownToken]:
            """Remove header tokens with placeholder content from markdown tokens."""
            filtered_tokens = []
            for token in markdown_tokens:
                if not (token['type'] == 'header' and token['content'].strip() == 'placeholder'):
                    filtered_tokens.append(token)
            return filtered_tokens

        def prepend_hashtags_to_headers(markdown_tokens: List[MarkdownToken]) -> List[MarkdownToken]:
            """Prepend hashtags to header tokens based on their level."""
            for token in markdown_tokens:
                if token['type'] == 'header' and token['level']:
                    # Add the appropriate number of hashtags based on header level
                    hashtags = '#' * token['level']
                    if not token['content'].startswith(hashtags):
                        token['content'] = f"{hashtags} {token['content']}"
            return markdown_tokens

        tokens = parse_with_timeout(md_content)
        tokens = remove_list_table_placeholders(tokens)
        tokens = prepend_hashtags_to_headers(tokens)
        if merge_contents:
            tokens = merge_tokens(tokens)
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


def convert_html_to_markdown(html_input: Union[str, Path], ignore_links: bool = True) -> str:
    """Convert HTML to Markdown with enhanced noise removal."""
    if isinstance(html_input, Path):
        with html_input.open('r', encoding='utf-8') as f:
            html_content = preprocess_html(f.read())
    else:
        html_content = preprocess_html(html_input)

    html_content = add_list_table_placeholders(html_content)

    converter = html2text.HTML2Text()
    converter.ignore_links = ignore_links
    converter.ignore_images = True
    converter.ignore_emphasis = True
    converter.mark_code = False
    converter.body_width = 0

    markdown_content = converter.handle(html_content)

    return markdown_content.strip()


def analyze_markdown(input: Union[str, Path]) -> MarkdownAnalysis:
    """
    Analyze markdown content and return structured analysis results.

    Args:
        input: Either a string containing markdown content or a Path to a markdown file.

    Returns:
        A dictionary containing detailed analysis of markdown elements.

    Raises:
        OSError: If the input file does not exist.
        TimeoutError: If analysis takes too long.
    """
    temp_md_path: Optional[Path] = None
    try:
        md_content = read_md_content(input)

        # Preprocess markdown
        md_content = preprocess_markdown(md_content)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as temp_file:
            temp_file.write(md_content)
            temp_md_path = Path(temp_file.name)

        # @timeout(3)
        def analyze_with_timeout(temp_file_path: str) -> tuple:
            analyzer = MarkdownAnalyzer(temp_file_path)
            raw_headers = convert_dict_keys_to_snake_case(
                analyzer.identify_headers())
            raw_paragraphs = convert_dict_keys_to_snake_case(
                analyzer.identify_paragraphs())
            raw_blockquotes = convert_dict_keys_to_snake_case(
                analyzer.identify_blockquotes())
            raw_code_blocks = convert_dict_keys_to_snake_case(
                analyzer.identify_code_blocks())
            raw_lists = convert_dict_keys_to_snake_case(
                analyzer.identify_lists())
            raw_tables = convert_dict_keys_to_snake_case(
                analyzer.identify_tables())
            raw_links = convert_dict_keys_to_snake_case(
                analyzer.identify_links())
            raw_footnotes = convert_dict_keys_to_snake_case(
                analyzer.identify_footnotes())
            raw_inline_code = convert_dict_keys_to_snake_case(
                analyzer.identify_inline_code())
            raw_emphasis = convert_dict_keys_to_snake_case(
                analyzer.identify_emphasis())
            raw_task_items = convert_dict_keys_to_snake_case(
                analyzer.identify_task_items())
            raw_html_blocks = convert_dict_keys_to_snake_case(
                analyzer.identify_html_blocks())
            raw_html_inline = convert_dict_keys_to_snake_case(
                analyzer.identify_html_inline())
            raw_tokens_sequential = convert_dict_keys_to_snake_case(
                analyzer.get_tokens_sequential())
            raw_summary = convert_dict_keys_to_snake_case(
                get_summary(md_content))
            return (
                raw_summary, raw_headers, raw_paragraphs, raw_blockquotes, raw_code_blocks,
                raw_lists, raw_tables, raw_links, raw_footnotes, raw_inline_code,
                raw_emphasis, raw_task_items, raw_html_blocks, raw_html_inline,
                raw_tokens_sequential, analyzer.count_words(), analyzer.count_characters()
            )

        (
            raw_summary, raw_headers, raw_paragraphs, raw_blockquotes, raw_code_blocks,
            raw_lists, raw_tables, raw_links, raw_footnotes, raw_inline_code,
            raw_emphasis, raw_task_items, raw_html_blocks, raw_html_inline,
            raw_tokens_sequential, word_count, char_count
        ) = analyze_with_timeout(str(temp_md_path))

        analysis_results: MarkdownAnalysis = {
            "summary": raw_summary,
            "headers": {
                "header": [
                    {**header, "text": clean_markdown_text(header["text"])}
                    for header in raw_headers.get("header", [])
                ]
            },
            "paragraphs": {
                "paragraph": [clean_markdown_text(p) for p in raw_paragraphs.get("paragraph", [])]
            },
            "blockquotes": {
                "blockquote": [clean_markdown_text(b) for b in raw_blockquotes.get("blockquote", [])]
            },
            "code_blocks": {
                "code_block": [
                    {**cb, "content": clean_markdown_text(cb["content"])}
                    for cb in raw_code_blocks.get("code_block", [])
                ]
            },
            "lists": {
                key: [
                    [{**item, "text": clean_markdown_text(item["text"])}
                     for item in sublist]
                    for sublist in v
                ]
                for key, v in raw_lists.items()
            },
            "tables": {
                "table": [
                    {
                        "header": [clean_markdown_text(h) for h in table["header"]],
                        "rows": [
                            [clean_markdown_text(cell) for cell in row]
                            for row in table["rows"]
                        ]
                    }
                    for table in raw_tables.get("table", [])
                ]
            },
            "links": {
                key: [
                    {
                        **link,
                        "text": clean_markdown_text(link.get("text")) if link.get("text") is not None else None,
                        "alt_text": clean_markdown_text(link.get("alt_text")) if link.get("alt_text") is not None else None,
                        "line": link["line"],
                        "url": link["url"]
                    }
                    for link in v
                ]
                for key, v in raw_links.items()
            },
            "footnotes": [
                {**fn, "content": clean_markdown_text(fn["content"])}
                for fn in raw_footnotes
            ],
            "inline_code": [
                {**ic, "code": clean_markdown_text(ic["code"])}
                for ic in raw_inline_code
            ],
            "emphasis": [
                {
                    "emphasis": {**em, "text": clean_markdown_text(em["text"])},
                    "text": clean_markdown_text(em["text"])
                }
                for em in raw_emphasis
            ],
            "task_items": [
                {**ti, "text": clean_markdown_text(ti["text"])}
                for ti in raw_task_items
            ],
            "html_blocks": [
                {**hb, "content": clean_markdown_text(hb["content"])}
                for hb in raw_html_blocks
            ],
            "html_inline": [
                {**hi, "html": clean_markdown_text(hi["html"])}
                for hi in raw_html_inline
            ],
            "tokens_sequential": [
                {**token, "content": clean_markdown_text(token["content"])}
                for token in raw_tokens_sequential
            ],
            "word_count": {"word_count": word_count},
            "char_count": {"char": char_count}
        }

        return analysis_results

    except TimeoutError as e:
        logger.error(f"Analysis timed out: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error analyzing markdown: {str(e)}")
        raise
    finally:
        if temp_md_path and temp_md_path.exists():
            try:
                temp_md_path.unlink()
            except OSError:
                logger.warning(
                    f"Warning: Could not delete temporary file {temp_md_path}")


def get_summary(input: Union[str, Path]) -> SummaryDict:
    temp_md_path: Optional[Path] = None
    try:
        md_content = read_md_content(input)
        md_content = preprocess_markdown(md_content)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as temp_file:
            temp_file.write(md_content)
            temp_md_path = Path(temp_file.name)
        analyzer = MarkdownAnalyzer(str(temp_md_path))

        # Get headers for counting individual levels
        headers = convert_dict_keys_to_snake_case(
            analyzer.identify_headers()).get("header", [])
        header_counts = {f"h{i}": 0 for i in range(1, 7)}
        for header in headers:
            level = header.get("level")
            if level in range(1, 7):
                header_counts[f"h{level}"] += 1
            else:
                logger.warning("Invalid header level detected: %s", level)

        # Get existing analysis
        raw_summary = convert_dict_keys_to_snake_case(analyzer.analyse())

        # Update summary with header counts
        summary = {
            **raw_summary,
            "header_counts": header_counts,
        }

        return summary
    finally:
        if temp_md_path and temp_md_path.exists():
            try:
                temp_md_path.unlink()
            except OSError:
                logger.warning(
                    f"Warning: Could not delete temporary file {temp_md_path}")


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
                line = f"{prefix_str} {checkbox}{' ' if checkbox else ''}{item['text']}".strip(
                )
                if token['type'] == 'unordered_list' and prefix_str == '*':
                    # Replace asterisk with dash for unordered lists
                    line = line.replace('* ', '- ')
                lines.append(line)
            result = '\n'.join(lines)

    elif token['type'] == 'table':
        if not token['meta'] or 'header' not in token['meta'] or 'rows' not in token['meta']:
            result = ""
        else:
            header = token['meta']['header']
            rows = token['meta']['rows']
            widths = [max(len(str(cell)) for row in [header] +
                          rows for cell in row[i:i+1]) for i in range(len(header))]
            lines = ['| ' + ' | '.join(cell.ljust(widths[i])
                                       for i, cell in enumerate(header)) + ' |']
            lines.append(
                '| ' + ' | '.join('-' * width for width in widths) + ' |')
            for row in rows:
                lines.append(
                    '| ' + ' | '.join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)) + ' |')
            result = '\n'.join(lines)

    elif token['type'] == 'code':
        content = token['content']
        # Remove code block delimiters and strip whitespace
        result = re.sub(r'^```[\w]*\n|\n```$', '', content).strip()

    else:  # paragraph, blockquote, html_block
        result = token['content']

    return result.strip()
