import os
import tempfile
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, TypedDict

import html2text
# import markdownify
from jet.code.html_utils import preprocess_html, valid_html
from jet.code.markdown_types import MarkdownAnalysis, MarkdownToken, SummaryDict
from jet.decorators.timer import timeout
from jet.transformers.object import convert_dict_keys_to_snake_case, make_serializable
from jet.utils.text import fix_and_unidecode
from mrkdwn_analysis import MarkdownAnalyzer, MarkdownParser

from jet.logger import logger


# def convert_html_to_markdown(html_input: Union[str, Path], ignore_links: bool = False) -> str:
#     """Convert HTML to Markdown with enhanced noise removal."""
#     logger.info("Starting HTML to Markdown conversion")
#     if isinstance(html_input, Path):
#         with html_input.open('r', encoding='utf-8') as f:
#             html_content = preprocess_html(f.read())
#     else:
#         html_content = preprocess_html(html_input)

#     html_content = add_list_table_header_placeholders(html_content)

#     try:
#         if isinstance(html_input, Path):
#             with html_input.open('r', encoding='utf-8') as f:
#                 html_content = f.read()
#         else:
#             html_content = html_input

#         # Convert to Markdown using markdownify
#         markdown_content = markdownify.MarkdownConverter(
#             autolinks=not ignore_links,
#             heading_style=markdownify.ATX
#         ).convert(html_content)

#         return markdown_content.strip()

#     except Exception as e:
#         logger.error("Failed to convert HTML to Markdown: %s", e)
#         raise

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

    markdown_content = converter.handle(html_content)

    return markdown_content.strip()


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
