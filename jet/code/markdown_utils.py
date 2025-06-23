import os
import tempfile
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, TypedDict

from markdownify import MarkdownConverter
from markitdown import MarkItDown
from jet.code.markdown_types import MarkdownAnalysis, MarkdownToken
from jet.decorators.timer import timeout
from mrkdwn_analysis import MarkdownAnalyzer, MarkdownParser

from jet.logger import logger


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


def convert_html_to_markdown(html_input: Union[str, Path], **options) -> str:
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
            result = md_converter.convert(str(html_input))
        else:
            # Write HTML content to a temporary file for MarkItDown
            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as temp_file:
                temp_file.write(html_input)
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


def parse_markdown(md_input: Union[str, Path]) -> List[MarkdownToken]:
    """
    Parse markdown content into a list of structured tokens using MarkdownParser.

    Args:
        md_input: Either a string containing markdown content or a Path to a markdown file.

    Returns:
        A list of dictionaries representing parsed markdown tokens with their type, content, and metadata.

    Raises:
        OSError: If the input file does not exist.
        TimeoutError: If parsing takes too long.
    """
    try:
        if isinstance(md_input, (str, Path)) and str(md_input).endswith('.md'):
            if not Path(str(md_input)).is_file():
                raise OSError(f"File {md_input} does not exist")
            with open(md_input, 'r', encoding='utf-8') as file:
                md_content = file.read()
        else:
            md_content = str(md_input)

        # Preprocess markdown
        logger.debug("Preprocessing markdown content")
        md_content = preprocess_markdown(md_content)

        # @timeout(3)
        def parse_with_timeout(content: str) -> List[MarkdownToken]:
            parser = MarkdownParser(content)
            return parser.parse()

        logger.debug("Starting markdown parsing")
        tokens = parse_with_timeout(md_content)
        parsed_tokens = [
            {
                "type": token.type,
                "content": clean_markdown_text(token.content),
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


def analyze_markdown(md_input: Union[str, Path]) -> MarkdownAnalysis:
    """
    Analyze markdown content and return structured analysis results.

    Args:
        md_input: Either a string containing markdown content or a Path to a markdown file.

    Returns:
        A dictionary containing detailed analysis of markdown elements.

    Raises:
        OSError: If the input file does not exist.
        TimeoutError: If analysis takes too long.
    """
    temp_md_path: Optional[Path] = None
    try:
        if isinstance(md_input, (str, Path)) and str(md_input).endswith('.md'):
            if not Path(str(md_input)).is_file():
                raise OSError(f"File {md_input} does not exist")
            with open(md_input, 'r', encoding='utf-8') as file:
                md_content = file.read()
        else:
            md_content = str(md_input)

        # Preprocess markdown
        logger.debug("Preprocessing markdown content")
        md_content = preprocess_markdown(md_content)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as temp_file:
            temp_file.write(md_content)
            temp_md_path = Path(temp_file.name)

        # @timeout(3)
        def analyze_with_timeout(temp_file_path: str) -> tuple:
            analyzer = MarkdownAnalyzer(temp_file_path)
            raw_headers = analyzer.identify_headers()
            raw_paragraphs = analyzer.identify_paragraphs()
            raw_blockquotes = analyzer.identify_blockquotes()
            raw_code_blocks = analyzer.identify_code_blocks()
            raw_lists = analyzer.identify_lists()
            raw_tables = analyzer.identify_tables()
            raw_links = analyzer.identify_links()
            raw_footnotes = analyzer.identify_footnotes()
            raw_inline_code = analyzer.identify_inline_code()
            raw_emphasis = analyzer.identify_emphasis()
            raw_task_items = analyzer.identify_task_items()
            raw_html_blocks = analyzer.identify_html_blocks()
            raw_html_inline = analyzer.identify_html_inline()
            raw_tokens_sequential = analyzer.get_tokens_sequential()
            raw_summary = get_summary(md_content)
            return (
                raw_summary, raw_headers, raw_paragraphs, raw_blockquotes, raw_code_blocks,
                raw_lists, raw_tables, raw_links, raw_footnotes, raw_inline_code,
                raw_emphasis, raw_task_items, raw_html_blocks, raw_html_inline,
                raw_tokens_sequential, analyzer.count_words(), analyzer.count_characters()
            )

        logger.debug("Starting markdown analysis")
        (
            raw_summary, raw_headers, raw_paragraphs, raw_blockquotes, raw_code_blocks,
            raw_lists, raw_tables, raw_links, raw_footnotes, raw_inline_code,
            raw_emphasis, raw_task_items, raw_html_blocks, raw_html_inline,
            raw_tokens_sequential, word_count, char_count
        ) = analyze_with_timeout(str(temp_md_path))

        analysis_results: MarkdownAnalysis = {
            "summary": raw_summary,
            "headers": {
                "Header": [
                    {**header, "text": clean_markdown_text(header["text"])}
                    for header in raw_headers.get("Header", [])
                ]
            },
            "paragraphs": {
                "Paragraph": [clean_markdown_text(p) for p in raw_paragraphs.get("Paragraph", [])]
            },
            "blockquotes": {
                "Blockquote": [clean_markdown_text(b) for b in raw_blockquotes.get("Blockquote", [])]
            },
            "code_blocks": {
                "Code block": [
                    {**cb, "content": clean_markdown_text(cb["content"])}
                    for cb in raw_code_blocks.get("Code block", [])
                ]
            },
            "lists": {
                k: [
                    [{**item, "text": clean_markdown_text(item["text"])}
                     for item in sublist]
                    for sublist in v
                ]
                for k, v in raw_lists.items()
            },
            "tables": {
                "Table": [
                    {
                        "header": [clean_markdown_text(h) for h in table["header"]],
                        "rows": [
                            [clean_markdown_text(cell) for cell in row]
                            for row in table["rows"]
                        ]
                    }
                    for table in raw_tables.get("Table", [])
                ]
            },
            "links": {
                k: [
                    {
                        **link,
                        "text": clean_markdown_text(link.get("text")),
                        "alt_text": clean_markdown_text(link.get("alt_text"))
                    }
                    for link in v
                ]
                for k, v in raw_links.items()
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
                    "Emphasis": em,
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


def get_summary(md_input: Union[str, Path]) -> MarkdownAnalysis:
    """
    Get summary analysis of markdown content using MarkdownAnalyzer.

    Args:
        md_input: Either a string containing markdown content or a Path to a markdown file.

    Returns:
        A dictionary containing summary analysis of markdown elements.

    Raises:
        OSError: If the input file does not exist.
    """
    temp_md_path: Optional[Path] = None
    try:
        if isinstance(md_input, (str, Path)) and str(md_input).endswith('.md'):
            if not Path(str(md_input)).is_file():
                raise OSError(f"File {md_input} does not exist")
            with open(md_input, 'r', encoding='utf-8') as file:
                md_content = file.read()
        else:
            md_content = str(md_input)

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
