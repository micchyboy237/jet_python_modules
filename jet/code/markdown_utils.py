import os
import tempfile
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, TypedDict

import html2text
from markdownify import MarkdownConverter
from markitdown import MarkItDown
from jet.code.html_utils import is_html
from jet.code.markdown_types import MarkdownAnalysis, MarkdownToken
from jet.decorators.timer import timeout
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
    # Process separator lines
    md_content = process_separator_lines(md_content)
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


def clean_markdown_text(text: Optional[str]) -> Optional[str]:
    """
    Clean markdown text by removing unnecessary escape characters.

    Args:
        text: The markdown text to clean, or None.

    Returns:
        The cleaned text with unnecessary escapes removed, or None if input is None.
    """
    if text is None:
        return None
    # Remove escaped periods (e.g., "\." -> ".")
    cleaned_text = re.sub(r'\\([.])', r'\1', text)
    return cleaned_text


def convert_html_to_markdownify(html_input: Union[str, Path], **options) -> str:
    """
    Convert HTML content to Markdown and return the string.
    """
    logger.info("Starting HTML to Markdown conversion")
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

        logger.debug("Markdown content generated: %s", markdown_content[:100])
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

        # Pre-process HTML to remove unwanted elements
        def remove_unwanted_elements(html: str) -> str:
            unwanted_elements = r'button|script|style|form|input|select|textarea'
            pattern = rf'<({unwanted_elements})(?:\s+[^>]*)?>.*?</\1>'
            return re.sub(pattern, '', html, flags=re.DOTALL)

        # Pre-process HTML to add whitespace between consecutive inline elements
        def add_space_between_inline_elements(html: str) -> str:
            inline_elements = r'span|a|strong|em|b|i|code|small|sub|sup|mark|del|ins|q'
            pattern = rf'</({inline_elements})><({inline_elements})'
            return re.sub(pattern, r'</\1> <\2', html)

        if isinstance(html_input, Path):
            # Read HTML file content and pre-process
            with open(html_input, 'r', encoding='utf-8') as file:
                html_content = file.read()
                html_content = remove_unwanted_elements(html_content)
                html_content = add_space_between_inline_elements(html_content)
            # Write pre-processed content to a temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as temp_file:
                temp_file.write(html_content)
                temp_file_path = temp_file.name
            try:
                result = md_converter.convert(temp_file_path)
            finally:
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
        else:
            # Pre-process HTML string and write to temporary file
            html_content = remove_unwanted_elements(html_input)
            html_content = add_space_between_inline_elements(html_content)
            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as temp_file:
                temp_file.write(html_content)
                temp_file_path = temp_file.name
            try:
                result = md_converter.convert(temp_file_path)
            finally:
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)

        markdown_content = result.text_content
        logger.debug("Markdown content generated: %s", markdown_content[:100])
        return markdown_content

    except Exception as e:
        logger.error("Failed to convert HTML to Markdown: %s", e)
        raise


def convert_html_to_markdown(html_input: Union[str, Path], ignore_links: bool = True) -> str:
    """Convert HTML to Markdown with enhanced noise removal."""
    def remove_unwanted_elements(html: str) -> str:
        unwanted_elements = r'button|script|style|form|input|select|textarea'
        pattern = rf'<({unwanted_elements})(?:\s+[^>]*)?>.*?</\1>'
        return re.sub(pattern, '', html, flags=re.DOTALL)

    def add_space_between_inline_elements(html: str) -> str:
        inline_elements = r'span|a|strong|em|b|i|code|small|sub|sup|mark|del|ins|q'
        pattern = rf'</({inline_elements})><({inline_elements})'
        return re.sub(pattern, r'</\1> <\2', html)

    if isinstance(html_input, Path):
        with html_input.open('r', encoding='utf-8') as f:
            html_content = f.read()
    else:
        html_content = html_input

    html_content = remove_unwanted_elements(html_content)
    html_content = add_space_between_inline_elements(html_content)

    converter = html2text.HTML2Text()
    converter.ignore_links = ignore_links
    converter.ignore_images = True
    converter.ignore_emphasis = True
    converter.mark_code = False
    converter.body_width = 0

    markdown_content = converter.handle(html_content)
    return markdown_content.strip()


def parse_markdown(input: Union[str, Path]) -> List[MarkdownToken]:
    def read_file(file_path: Path) -> str:
        if not file_path.is_file():
            raise OSError(f"File {file_path} does not exist")
        with file_path.open('r', encoding='utf-8') as file:
            return file.read()

    def parse_with_timeout(content: str) -> List[MarkdownToken]:
        parser = MarkdownParser(content)
        return parser.parse()

    try:
        input_str = str(input)
        input_path = Path(input_str)

        if is_html(input_str) or input_str.endswith('.html'):
            md_content = convert_html_to_markdown(input)
        elif input_str.endswith('.md'):
            md_content = read_file(input_path)
        else:
            md_content = input_str

        logger.debug("Preprocessing markdown content")
        md_content = preprocess_markdown(md_content)

        logger.debug("Starting markdown parsing")
        tokens = parse_with_timeout(md_content)

        parsed_tokens = [
            {
                "type": token.type,
                "content": derive_text({
                    "type": token.type,
                    "content": token.content,
                    "level": token.level,
                    "meta": token.meta,
                    "line": token.line
                }),
                "level": token.level,
                "meta": token.meta,
                "line": token.line
            }
            for token in tokens
        ]

        logger.debug(f"Parsed {len(parsed_tokens)} markdown tokens")
        return parsed_tokens

    except TimeoutError as e:
        logger.error(f"Parsing timed out: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error parsing markdown: {str(e)}")
        raise


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
    def read_file(file_path: Path) -> str:
        if not file_path.is_file():
            raise OSError(f"File {file_path} does not exist")
        return file_path.read_text(encoding='utf-8')

    def analyze_with_timeout(temp_file_path: str) -> tuple:
        analyzer = MarkdownAnalyzer(temp_file_path)
        return (
            convert_dict_keys_to_snake_case(get_summary(md_content)),
            convert_dict_keys_to_snake_case(analyzer.identify_headers()),
            convert_dict_keys_to_snake_case(analyzer.identify_paragraphs()),
            convert_dict_keys_to_snake_case(analyzer.identify_blockquotes()),
            convert_dict_keys_to_snake_case(analyzer.identify_code_blocks()),
            convert_dict_keys_to_snake_case(analyzer.identify_lists()),
            convert_dict_keys_to_snake_case(analyzer.identify_tables()),
            convert_dict_keys_to_snake_case(analyzer.identify_links()),
            convert_dict_keys_to_snake_case(analyzer.identify_footnotes()),
            convert_dict_keys_to_snake_case(analyzer.identify_inline_code()),
            convert_dict_keys_to_snake_case(analyzer.identify_emphasis()),
            convert_dict_keys_to_snake_case(analyzer.identify_task_items()),
            convert_dict_keys_to_snake_case(analyzer.identify_html_blocks()),
            convert_dict_keys_to_snake_case(analyzer.identify_html_inline()),
            convert_dict_keys_to_snake_case(analyzer.get_tokens_sequential()),
            analyzer.count_words(),
            analyzer.count_characters()
        )

    temp_md_path: Optional[Path] = None

    try:
        input_str = str(input)
        input_path = Path(input_str)

        if input_str.endswith('.md'):
            md_content = read_file(input_path)
        else:
            md_content = input_str

        logger.debug("Preprocessing markdown content")
        md_content = preprocess_markdown(md_content)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as temp_file:
            temp_file.write(md_content)
            temp_md_path = Path(temp_file.name)

        logger.debug("Starting markdown analysis")
        (
            raw_summary, raw_headers, raw_paragraphs, raw_blockquotes, raw_code_blocks,
            raw_lists, raw_tables, raw_links, raw_footnotes, raw_inline_code,
            raw_emphasis, raw_task_items, raw_html_blocks, raw_html_inline,
            raw_tokens_sequential, word_count, char_count
        ) = analyze_with_timeout(str(temp_md_path))

        return {
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


def get_summary(input: Union[str, Path]) -> MarkdownAnalysis:
    """
    Get summary analysis of markdown content using MarkdownAnalyzer.

    Args:
        input: Either a string containing markdown content or a Path to a markdown file.

    Returns:
        A dictionary containing summary analysis of markdown elements.

    Raises:
        OSError: If the input file does not exist.
    """
    def read_file(file_path: Path) -> str:
        if not file_path.is_file():
            raise OSError(f"File {file_path} does not exist")
        return file_path.read_text(encoding='utf-8')

    temp_md_path: Optional[Path] = None

    try:
        input_str = str(input)
        input_path = Path(input_str)

        if input_str.endswith('.md'):
            md_content = read_file(input_path)
        else:
            md_content = input_str

        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as temp_file:
            temp_file.write(md_content)
            temp_md_path = Path(temp_file.name)

        analyzer = MarkdownAnalyzer(str(temp_md_path))
        return analyzer.analyse()

    finally:
        if temp_md_path and temp_md_path.exists():
            try:
                temp_md_path.unlink()
            except OSError:
                logger.warning(
                    f"Warning: Could not delete temporary file {temp_md_path}")


def derive_text(token: MarkdownToken) -> str:
    if token['type'] == 'header' and token['level'] is not None:
        return f"{'#' * token['level']} {token['content']}"
    elif token['type'] in ['unordered_list', 'ordered_list']:
        if not token['meta'] or 'items' not in token['meta']:
            return ""
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
            lines.append(line)
        return '\n'.join(lines)
    elif token['type'] == 'table':
        if not token['meta'] or 'header' not in token['meta'] or 'rows' not in token['meta']:
            return ""
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
        return '\n'.join(lines)
    elif token['type'] == 'code':
        language = token['meta'].get(
            'language', '') if token['meta'] else ''
        return f"```{' ' + language if language else ''}\n{token['content']}\n```"
    else:
        return token['content']
