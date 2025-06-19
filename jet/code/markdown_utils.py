import os
import tempfile
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, TypedDict

from markdownify import markdownify
from jet.code.markdown_types import MarkdownAnalysis, MarkdownToken
from mrkdwn_analysis import MarkdownAnalyzer, MarkdownParser

from jet.logger import logger


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
        markdown_content = markdownify(html_content, heading_style="ATX")
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
                "content": token.content,
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
        analysis_results: MarkdownAnalysis = {
            "headers": analyzer.identify_headers(),
            "paragraphs": analyzer.identify_paragraphs(),
            "blockquotes": analyzer.identify_blockquotes(),
            "code_blocks": analyzer.identify_code_blocks(),
            "lists": analyzer.identify_lists(),
            "tables": analyzer.identify_tables(),
            "links": analyzer.identify_links(),
            "footnotes": analyzer.identify_footnotes(),
            "inline_code": analyzer.identify_inline_code(),
            "emphasis": analyzer.identify_emphasis(),
            "task_items": analyzer.identify_task_items(),
            "html_blocks": analyzer.identify_html_blocks(),
            "html_inline": analyzer.identify_html_inline(),
            "tokens_sequential": analyzer.get_tokens_sequential(),
            "word_count": {"word_count": analyzer.count_words()},
            "char_count": [analyzer.count_characters()],
            "summary": analyzer.analyse(),
        }
        return analysis_results
    finally:
        if temp_md_path and temp_md_path.exists():
            try:
                temp_md_path.unlink()
            except OSError:
                print(
                    f"Warning: Could not delete temporary file {temp_md_path}")
