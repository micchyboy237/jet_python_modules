import os
import tempfile
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, TypedDict

from markdownify import MarkdownConverter
from jet.code.markdown_types import MarkdownAnalysis, MarkdownToken
from mrkdwn_analysis import MarkdownAnalyzer, MarkdownParser

from jet.logger import logger


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


def convert_html_to_markdown(html_input: Union[str, Path], **options) -> str:
    """
    Convert HTML content to Markdown and save to a file.

    Args:
        html_input: HTML content as a string or path to an HTML file.
        output_md_path: Path to save the generated Markdown file.
    """
    logger.info("Starting HTML to Markdown conversion")
    try:
        # Read HTML content
        if isinstance(html_input, Path):
            with html_input.open('r', encoding='utf-8') as f:
                html_content = f.read()
        else:
            html_content = html_input

        # Convert to Markdown
        markdown_content = MarkdownConverter(
            heading_style="ATX").convert(html_content)
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
    """
    try:
        if isinstance(md_input, (str, Path)) and str(md_input).endswith('.md'):
            if not Path(str(md_input)).is_file():
                raise OSError(f"File {md_input} does not exist")
            with open(md_input, 'r', encoding='utf-8') as file:
                md_content = file.read()
        else:
            md_content = str(md_input)

        parser = MarkdownParser(md_content)
        tokens = parser.parse()
        parsed_tokens = [
            {
                "type": token.type,
                # Apply text cleaning
                "content": clean_markdown_text(token.content),
                "level": token.level,
                "meta": token.meta,
                "line": token.line
            }
            for token in tokens
        ]
        logger.debug(f"Parsed {len(parsed_tokens)} markdown tokens")
        return parsed_tokens

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

        # Get raw analysis results
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
        raw_summary = analyzer.analyse()

        # Clean text fields in analysis results
        analysis_results: MarkdownAnalysis = {
            "summary": raw_summary,  # Summary contains counts, no text fields to clean
            "headers": {
                "Header": [
                    {**header, "text": clean_markdown_text(header["text"])}
                    for header in raw_headers["Header"]
                ]
            },
            "paragraphs": {
                "Paragraph": [clean_markdown_text(p) for p in raw_paragraphs["Paragraph"]]
            },
            "blockquotes": {
                "Blockquote": [clean_markdown_text(b) for b in raw_blockquotes["Blockquote"]]
            },
            "code_blocks": {
                "Code block": [
                    {**cb, "content": clean_markdown_text(cb["content"])}
                    for cb in raw_code_blocks["Code block"]
                ]
            },
            "lists": raw_lists,  # ListItem.text is cleaned below
            "tables": raw_tables,  # TableItem.header and rows are cleaned below
            "links": {
                k: [
                    {**link, "text": clean_markdown_text(
                        link.get("text")), "alt_text": clean_markdown_text(link.get("alt_text"))}
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
                {**em, "text": clean_markdown_text(em["text"])}
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
            "word_count": {"word_count": analyzer.count_words()},
            "char_count": [analyzer.count_characters()]
        }

        # Clean nested ListItem.text in lists
        analysis_results["lists"] = {
            k: [
                [{**item, "text": clean_markdown_text(item["text"])}
                 for item in sublist]
                for sublist in v
            ]
            for k, v in raw_lists.items()
        }

        # Clean nested TableItem.header and rows
        analysis_results["tables"] = {
            "Table": [
                {
                    "header": [clean_markdown_text(h) for h in table["header"]],
                    "rows": [[clean_markdown_text(cell) for cell in row] for row in table["rows"]]
                }
                for table in raw_tables["Table"]
            ]
        }

        return analysis_results
    finally:
        if temp_md_path and temp_md_path.exists():
            try:
                temp_md_path.unlink()
            except OSError:
                print(
                    f"Warning: Could not delete temporary file {temp_md_path}")
