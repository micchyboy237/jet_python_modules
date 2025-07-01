import os
import tempfile
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, TypedDict

import html2text
from jet.code.html_utils import preprocess_html, valid_html
from jet.code.markdown_types import MarkdownAnalysis, MarkdownToken, SummaryDict
from jet.code.markdown_utils import convert_html_to_markdown, clean_markdown_links
from jet.code.markdown_utils._converters import convert_markdown_to_html
from jet.decorators.timer import timeout
from jet.transformers.object import convert_dict_keys_to_snake_case, make_serializable
from jet.utils.text import fix_and_unidecode
from mrkdwn_analysis import MarkdownAnalyzer, MarkdownParser

from jet.logger import logger


def read_md_content(input, ignore_links: bool = False) -> str:
    md_content = ""
    try:
        if Path(str(input)).is_file():
            with open(input, 'r', encoding='utf-8') as file:
                md_content = file.read()

            if str(input).endswith('.html'):
                md_content = convert_html_to_markdown(
                    md_content, ignore_links=ignore_links)
    except OSError:
        md_content = str(input)

        if valid_html(md_content):
            md_content = convert_html_to_markdown(
                md_content, ignore_links=ignore_links)
    except Exception:
        raise

    if ignore_links:
        md_content = clean_markdown_links(md_content)

    return md_content


def read_html_content(input, ignore_links: bool = False) -> str:
    """
    Reads HTML content from a file path or string input, optionally removing links.

    Args:
        input: Path to the HTML file or a string containing HTML content.
        ignore_links: If True, links will be removed from the resulting markdown.

    Returns:
        The HTML content as a string.
    """
    html_content = ""
    try:
        if Path(str(input)).is_file():
            with open(input, 'r', encoding='utf-8') as file:
                html_content = file.read()
        else:
            html_content = str(input)
    except Exception:
        html_content = str(input)

    # Optionally preprocess HTML (e.g., clean, normalize)
    html_content = preprocess_html(html_content)

    if ignore_links:
        # Convert to markdown, clean links, then convert back to HTML
        md = convert_markdown_to_html(html_content, ignore_links=True)
        # Optionally, convert back to HTML if needed, or just return the markdown
        # For now, return the markdown with links removed
        return md

    return html_content
