import os
import tempfile
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, TypedDict

import html2text
from jet.code.html_utils import preprocess_html, valid_html
from jet.code.markdown_types import MarkdownAnalysis, MarkdownToken, SummaryDict
from jet.code.markdown_types.converter_types import MarkdownExtensions
from jet.code.markdown_utils._preprocessors import clean_markdown_links
from jet.decorators.timer import timeout
from jet.transformers.object import convert_dict_keys_to_snake_case, make_serializable
from jet.utils.text import fix_and_unidecode
import markdown

from jet.logger import logger


def convert_html_to_markdown(html_input: Union[str, Path], ignore_links: bool = False) -> str:
    """Convert HTML to Markdown with enhanced noise removal."""
    if isinstance(html_input, Path):
        with html_input.open('r', encoding='utf-8') as f:
            html_content = preprocess_html(f.read())
    else:
        html_content = preprocess_html(html_input)

    html_content = add_list_table_header_placeholders(html_content)

    converter = html2text.HTML2Text()
    converter.ignore_links = ignore_links
    converter.ignore_images = True
    converter.ignore_emphasis = True
    converter.mark_code = False
    converter.body_width = 0

    md_content = converter.handle(html_content)

    md_content = fix_and_unidecode(md_content)

    return md_content.strip()


def convert_markdown_to_html(md_content: str, exts: MarkdownExtensions = {"extensions": [
    "extra", "fenced_code", "tables", "sane_lists", "toc"
]}) -> str:
    """
    Convert markdown to HTML with supported extensions enabled.

    Args:
        md_content (str): Markdown content to convert.
        exts (MarkdownExtensions): Dictionary containing a list of extension names.

    Returns:
        str: Rendered HTML output.
    """
    # Map extension names to their corresponding markdown extension paths
    extension_map = {
        "extra": "markdown.extensions.extra",
        "abbr": "markdown.extensions.abbr",
        "attr_list": "markdown.extensions.attr_list",
        "def_list": "markdown.extensions.def_list",
        "fenced_code": "markdown.extensions.fenced_code",
        "footnotes": "markdown.extensions.footnotes",
        "md_in_html": "markdown.extensions.md_in_html",
        "tables": "markdown.extensions.tables",
        "admonition": "markdown.extensions.admonition",
        "codehilite": "markdown.extensions.codehilite",
        "legacy_attrs": "markdown.extensions.legacy_attrs",
        "legacy_em": "markdown.extensions.legacy_em",
        "meta": "markdown.extensions.meta",
        "nl2br": "markdown.extensions.nl2br",
        "sane_lists": "markdown.extensions.sane_lists",
        "smarty": "markdown.extensions.smarty",
        "toc": "markdown.extensions.toc",
        "wikilinks": "markdown.extensions.wikilinks",
    }

    # Validate and collect extensions
    valid_extensions = []
    for ext in exts.get("extensions", []):
        if ext in extension_map:
            valid_extensions.append(extension_map[ext])
        else:
            raise ValueError(f"Unsupported markdown extension: {ext}")

    # Debug log: Log the extensions being used
    logger.debug(f"Rendering markdown with extensions: {valid_extensions}")

    md_content = fix_and_unidecode(md_content)

    # Render markdown with specified extensions
    html = markdown.markdown(md_content, extensions=valid_extensions)

    # Debug log: Log the rendered HTML
    logger.debug(f"Rendered HTML: {html}")

    return html


def add_list_table_header_placeholders(html: str) -> str:
    """
    Add <h6> placeholders after </ol>, </ul>, and </table> tags to prevent markdown parser issues.

    Args:
        html: Input HTML string to process.

    Returns:
        HTML string with <h6> placeholders added after specified tags.
    """
    return re.sub(r'</(ol|ul|table)>', r'</\1><h1>placeholder</h1>', html, flags=re.IGNORECASE)


__all__ = [
    "convert_html_to_markdown",
]
