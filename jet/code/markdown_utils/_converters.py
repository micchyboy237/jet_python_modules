import os
import tempfile
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, TypedDict

import html2text
from jet.code.html_utils import format_html, preprocess_html, valid_html
from jet.code.markdown_types import MarkdownAnalysis, MarkdownToken, SummaryDict
from jet.code.markdown_types.converter_types import MarkdownExtensions
from jet.code.markdown_utils._preprocessors import clean_markdown_links, preprocess_markdown
from jet.decorators.timer import timeout
from jet.file.utils import load_file
from jet.transformers.object import convert_dict_keys_to_snake_case, make_serializable
from jet.utils.text import fix_and_unidecode
import markdown

from jet.logger import logger


def convert_html_to_markdown(html_input: Union[str, Path], ignore_links: bool = False) -> str:
    """Convert HTML to Markdown with enhanced noise removal."""
    html_content: str
    if isinstance(html_input, (str, Path)) and str(html_input).endswith(('.html', '.htm')) and Path(str(html_input)).is_file():
        html_content = load_file(str(html_input))
        html_content = preprocess_html(html_content)
    else:
        html_content = preprocess_html(str(html_input))

    html_content = format_html(html_content)

    converter = html2text.HTML2Text()
    converter.ignore_links = ignore_links
    converter.ignore_images = True
    converter.ignore_emphasis = True
    converter.mark_code = False
    converter.body_width = 0

    md_content = converter.handle(html_content)

    md_content = preprocess_markdown(md_content)

    return md_content.strip()


def convert_markdown_to_html(md_input: Union[str, Path], exts: MarkdownExtensions = {"extensions": [
    "fenced_code", "tables", "sane_lists", "toc", "attr_list", "def_list"
]}, ignore_links: bool = False) -> str:
    """
    Convert markdown to HTML with supported extensions enabled.

    Args:
        md_input (Union[str, Path]): Markdown content or path to convert.
        exts (MarkdownExtensions): Dictionary containing a list of extension names.

    Returns:
        str: Rendered HTML output.
    """

    if isinstance(md_input, Path):
        with md_input.open('r', encoding='utf-8') as f:
            md_content = preprocess_html(f.read())
    else:
        md_content = preprocess_html(md_input)

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

    md_content = preprocess_markdown(md_content)

    if ignore_links:
        md_content = clean_markdown_links(md_content)

    # Render markdown with specified extensions
    html_content = markdown.markdown(md_content, extensions=valid_extensions)

    html_content = format_html(html_content)

    return html_content


__all__ = [
    "convert_html_to_markdown",
]
