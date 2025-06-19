import logging
from pathlib import Path
import tempfile
from typing import Dict, List, Optional, TypedDict, Union
from markdownify import markdownify
from mrkdwn_analysis import MarkdownAnalyzer

from jet.file.utils import save_file

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MarkdownAnalysisResult(TypedDict):
    headers: int
    paragraphs: int
    links: int
    code_blocks: int


def convert_html_to_markdown(html_input: Union[str, Path]) -> str:
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


def analyze_markdown_file(md_path: str | Path) -> MarkdownAnalysisResult:
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
        analyzer = MarkdownAnalyzer(str(md_path))
        analysis = analyzer.analyse()

        result: MarkdownAnalysisResult = {
            'headers': analysis.get('headers', 0),
            'paragraphs': analysis.get('paragraphs', 0),
            'links': analysis.get('links', 0),
            'code_blocks': analysis.get('code_blocks', 0)
        }
        logger.info("Analysis result: %s", result)
        return result

    except Exception as e:
        logger.error("Failed to analyze Markdown: %s", e)
        raise


def process_html_for_analysis(html_input: Union[str, Path], output_md_path: Optional[str | Path] = None) -> MarkdownAnalysisResult:
    """
    Process HTML input to generate and analyze Markdown content.

    Args:
        html_input: HTML content as a string or path to an HTML file.
        output_md_path: str | Path to save the generated Markdown file.

    Returns:
        Analysis results from the Markdown file.
    """
    if output_md_path and not isinstance(output_md_path, Path):
        output_md_path = Path(output_md_path)

    md_content = convert_html_to_markdown(html_input)
    if not output_md_path:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as temp_file:
            temp_file.write(md_content)
            final_md_path = Path(temp_file.name)
    else:
        final_md_path = output_md_path
        save_file(md_content, str(final_md_path))

    try:
        return analyze_markdown_file(final_md_path)
    finally:
        # Safely remove the temporary file
        if not output_md_path and final_md_path.exists():
            try:
                final_md_path.unlink()
            except PermissionError:
                print(
                    f"Warning: Could not delete temporary file {final_md_path}")
