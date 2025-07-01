import os
import tempfile
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, TypedDict

import html2text
from jet.code.html_utils import preprocess_html, valid_html
from jet.code.markdown_types import MarkdownAnalysis, MarkdownToken, SummaryDict
from jet.code.markdown_utils import convert_html_to_markdown
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
    return md_content


# @timeout(3)
def base_parse_markdown(input: Union[str, Path], ignore_links: bool = False) -> List[MarkdownToken]:
    md_content = read_md_content(input, ignore_links=ignore_links)
    parser = MarkdownParser(md_content)
    return make_serializable(parser.parse())


# @timeout(3)
def base_analyze_markdown(input: Union[str, Path], ignore_links: bool = False) -> tuple:
    import tempfile
    import os

    md_content = read_md_content(input, ignore_links=ignore_links)
    temp_md_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".md", mode="w", encoding="utf-8") as tmpfile:
            tmpfile.write(md_content)
            temp_md_path = tmpfile.name

        analyzer = MarkdownAnalyzer(temp_md_path)
        raw_headers = convert_dict_keys_to_snake_case(
            analyzer.identify_headers())
        raw_paragraphs = convert_dict_keys_to_snake_case(
            analyzer.identify_paragraphs())
        raw_blockquotes = convert_dict_keys_to_snake_case(
            analyzer.identify_blockquotes())
        raw_code_blocks = convert_dict_keys_to_snake_case(
            analyzer.identify_code_blocks())
        raw_lists = convert_dict_keys_to_snake_case(
            analyzer.identify_lists())
        raw_tables = convert_dict_keys_to_snake_case(
            analyzer.identify_tables())
        raw_links = convert_dict_keys_to_snake_case(
            analyzer.identify_links())
        raw_footnotes = convert_dict_keys_to_snake_case(
            analyzer.identify_footnotes())
        raw_inline_code = convert_dict_keys_to_snake_case(
            analyzer.identify_inline_code())
        raw_emphasis = convert_dict_keys_to_snake_case(
            analyzer.identify_emphasis())
        raw_task_items = convert_dict_keys_to_snake_case(
            analyzer.identify_task_items())
        raw_html_blocks = convert_dict_keys_to_snake_case(
            analyzer.identify_html_blocks())
        raw_html_inline = convert_dict_keys_to_snake_case(
            analyzer.identify_html_inline())
        raw_tokens_sequential = convert_dict_keys_to_snake_case(
            analyzer.get_tokens_sequential())
        raw_word_count = {
            "word_count": analyzer.count_words()
        }
        raw_char_count = {
            "char_count": analyzer.count_characters()
        }

        # Get headers for counting individual levels
        headers = raw_headers.get("header", [])
        header_counts = {f"h{i}": 0 for i in range(1, 7)}
        for header in headers:
            level = header.get("level")
            if level in range(1, 7):
                header_counts[f"h{level}"] += 1
            else:
                logger.warning("Invalid header level detected: %s", level)

        # Get links for counting
        text_links = raw_links.get("text_link", [])
        image_links = raw_links.get("image_link", [])
        text_link_count = len(text_links)
        image_link_count = len(image_links)

        # Get existing analysis
        raw_summary = convert_dict_keys_to_snake_case(analyzer.analyse())

        # Update summary with header counts and link counts
        summary = {
            **raw_summary,
            "header_counts": header_counts,
            "text_links": text_link_count,
            "image_links": image_link_count,
        }

        # return {
        #     "summary": summary,
        #     "headers": headers,
        # }
        return {
            "raw_headers": raw_headers,
            "raw_paragraphs": raw_paragraphs,
            "raw_blockquotes": raw_blockquotes,
            "raw_code_blocks": raw_code_blocks,
            "raw_lists": raw_lists,
            "raw_tables": raw_tables,
            "raw_links": raw_links,
            "raw_footnotes": raw_footnotes,
            "raw_inline_code": raw_inline_code,
            "raw_emphasis": raw_emphasis,
            "raw_task_items": raw_task_items,
            "raw_html_blocks": raw_html_blocks,
            "raw_html_inline": raw_html_inline,
            "raw_tokens_sequential": raw_tokens_sequential,
            "raw_word_count": raw_word_count,
            "raw_char_count": raw_char_count,
            "raw_summary": raw_summary,
        }
    finally:
        if temp_md_path and os.path.exists(temp_md_path):
            try:
                os.remove(temp_md_path)
            except Exception as e:
                logger.warning(
                    f"Could not remove temporary file {temp_md_path}: {e}")
