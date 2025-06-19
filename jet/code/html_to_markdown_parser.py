import logging
from pathlib import Path
import tempfile
from typing import Dict, List, Optional, TypedDict, Union

from jet.code.markdown_utils import convert_html_to_markdown, parse_markdown, MarkdownToken
from jet.file.utils import save_file
from jet.logger import logger


def parse_markdown_file(md_path: str | Path) -> List[MarkdownToken]:
    """
    Analyze a Markdown file using mrkdwn_analysis and return key metrics.

    Args:
        md_path: str | Path to the Markdown file.

    Returns:
        A dictionary containing counts of headers, paragraphs, links, and code blocks.
    """
    if not isinstance(md_path, Path):
        md_path = Path(md_path)

    logger.info("Analyzing Markdown file: %s", md_path)
    try:
        result = parse_markdown(str(md_path))
        return result

    except Exception as e:
        logger.error("Failed to parse Markdown: %s", e)
        raise


def process_html_for_parsing(html_input: Union[str, Path], output_md_path: Optional[str | Path] = None) -> List[MarkdownToken]:
    """
    Process HTML input to generate and parse Markdown content.

    Args:
        html_input: HTML content as a string or path to an HTML file.
        output_md_path: str | Path to save the generated Markdown file.

    Returns:
        Analysis results from the Markdown file.
    """
    if output_md_path and not isinstance(output_md_path, Path):
        output_md_path = Path(output_md_path)

    try:
        md_content = convert_html_to_markdown(html_input)
        result = parse_markdown(md_content)

        if output_md_path:
            final_md_path = output_md_path
            save_file(md_content, str(final_md_path))

        return result
    finally:
        # Safely remove the temporary file
        if not output_md_path and final_md_path.exists():
            try:
                final_md_path.unlink()
            except PermissionError:
                print(
                    f"Warning: Could not delete temporary file {final_md_path}")
