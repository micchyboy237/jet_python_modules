import os
import tempfile
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, TypedDict

import html2text
from jet.code.html_utils import preprocess_html, valid_html
from jet.code.markdown_types import MarkdownAnalysis, MarkdownToken, SummaryDict
from jet.code.markdown_utils import convert_html_to_markdown, clean_markdown_links
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

    if not ignore_links:
        md_content = clean_markdown_links(md_content)

    return md_content
