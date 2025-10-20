import re

import html2text
import markdown
from markdown_it import MarkdownIt
from mdit_plain.renderer import RendererPlain
from pathlib import Path
from typing import Union
from jet.code.html_utils import preprocess_html
from jet.transformers.formatters import format_html
from jet.code.markdown_types.converter_types import MarkdownExtensions
from jet.code.markdown_utils._preprocessors import remove_markdown_links, preprocess_markdown
from jet.file.utils import load_file


def convert_html_to_markdown(html_input: Union[str, Path], ignore_links: bool = False) -> str:
    """Convert HTML to Markdown with enhanced noise removal."""
    html_content: str
    if isinstance(html_input, (str, Path)) and str(html_input).endswith(('.html', '.htm')) and Path(str(html_input)).is_file():
        html_content = load_file(str(html_input))
        html_content = preprocess_html(html_content, excludes=["script", "style"])
    else:
        html_content = preprocess_html(str(html_input), excludes=["script", "style"])

    html_content = format_html(html_content)

    # # Add header placeholder after closing list elements (ul, ol, table)
    # html_content = add_list_table_header_placeholders(html_content)

    converter = html2text.HTML2Text()
    converter.ignore_links = False
    converter.ignore_mailto_links = ignore_links
    # converter.protect_links = False
    converter.ignore_images = True
    converter.ignore_emphasis = True
    converter.mark_code = True
    converter.body_width = 0

    md_content = converter.handle(html_content)
    if ignore_links:
        md_content = remove_markdown_links(md_content)
    md_content = preprocess_markdown(md_content)

    # preprocessed_html_content = convert_markdown_to_html(md_content)

    # # Remove placeholder headers
    # preprocessed_html_content = re.sub(
    #     r'^\s*#{1,6}\s*Placeholder\s*$', '', preprocessed_html_content, flags=re.MULTILINE)

    # preprocessed_md_content = converter.handle(preprocessed_html_content)

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

    if ignore_links:
        md_content = remove_markdown_links(md_content)

    md_content = preprocess_markdown(md_content)

    # Render markdown with specified extensions
    html_content = markdown.markdown(md_content, extensions=valid_extensions)

    html_content = format_html(html_content)

    return html_content


def convert_markdown_to_text(md_content: str) -> str:
    """
    Convert Markdown string to plain text using mdit-plain.

    Args:
        md_content (str): Input Markdown string.

    Returns:
        str: Plain text string.

    Raises:
        ValueError: If md_content is empty or not a string.
    """
    if not isinstance(md_content, str):
        raise ValueError("Input must be a string")
    if not md_content.strip():
        raise ValueError("Input Markdown content cannot be empty")

    parser = MarkdownIt(renderer_cls=RendererPlain)
    plain_text = parser.render(md_content)
    return plain_text


def add_list_table_header_placeholders(html: str) -> str:
    """
    Add <h1> placeholders after </ol>, </ul>, and </table> tags to prevent markdown parser issues.

    Args:
        html: Input HTML string to process.

    Returns:
        HTML string with <h1> placeholders added after specified tags.
    """
    return re.sub(r'</(ol|ul|table)>', r'</\1><h1>placeholder</h1>', html, flags=re.IGNORECASE)


__all__ = [
    "convert_html_to_markdown",
    "convert_markdown_to_html",
    "convert_markdown_to_text",
]
