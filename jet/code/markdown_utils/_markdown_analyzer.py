import os
import tempfile
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, TypedDict

import html2text
from jet.code.html_utils import preprocess_html, valid_html
from jet.code.markdown_types import MarkdownAnalysis, MarkdownToken, SummaryDict
from jet.code.markdown_utils import read_md_content, preprocess_markdown
from jet.decorators.timer import timeout
from jet.transformers.object import convert_dict_keys_to_snake_case, make_serializable
from jet.utils.text import fix_and_unidecode
from mrkdwn_analysis import MarkdownAnalyzer, MarkdownParser

from jet.logger import logger

DEFAULTS: MarkdownAnalysis = {
    "summary": {
        "headers": 0,
        "header_counts": {"h1": 0, "h2": 0, "h3": 0, "h4": 0, "h5": 0, "h6": 0},
        "paragraphs": 0,
        "blockquotes": 0,
        "code_blocks": 0,
        "ordered_lists": 0,
        "unordered_lists": 0,
        "tables": 0,
        "html_blocks": 0,
        "html_inline_count": 0,
        "words": 0,
        "characters": 0,
        "text_links": 0,
        "image_links": 0,
    },
    "word_count": 0,
    "char_count": 0,
    "headers": [],
    "paragraphs": [],
    "blockquotes": [],
    "code_blocks": [],
    "unordered_lists": [],
    "ordered_lists": [],
    "tables": [],
    "text_links": [],
    "image_links": [],
    "footnotes": [],
    "inline_code": [],
    "emphasis": [],
    "task_items": [],
    "html_blocks": [],
    "html_inline": [],
    "tokens_sequential": [],
}

# @timeout(3)


def base_analyze_markdown(input: Union[str, Path], ignore_links: bool = False) -> MarkdownAnalysis:
    import tempfile
    import os
    from jet.logger import logger
    from jet.transformers.object import convert_dict_keys_to_snake_case, make_serializable

    logger.debug(
        "Starting base_analyze_markdown with input: %s, ignore_links: %s", input, ignore_links)
    md_content = read_md_content(input, ignore_links=ignore_links)
    logger.debug("Markdown content after read_md_content: %s",
                 md_content[:100] + "..." if len(md_content) > 100 else md_content)
    temp_md_path = None
    values: MarkdownAnalysis = DEFAULTS.copy()
    try:
        md_content = preprocess_markdown(md_content)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".md", mode="w", encoding="utf-8") as tmpfile:
            tmpfile.write(md_content)
            temp_md_path = tmpfile.name
        logger.debug("Temporary markdown file created at: %s", temp_md_path)
        analyzer = MarkdownAnalyzer(temp_md_path)

        # Flatten dictionaries by extracting nested lists
        try:
            raw_headers = convert_dict_keys_to_snake_case(
                analyzer.identify_headers()).get("header", [])
            logger.debug("Headers identified: %s", raw_headers)
            raw_paragraphs = convert_dict_keys_to_snake_case(
                analyzer.identify_paragraphs()).get("paragraph", [])
            logger.debug("Paragraphs identified: %s", raw_paragraphs)
            raw_blockquotes = convert_dict_keys_to_snake_case(
                analyzer.identify_blockquotes()).get("blockquote", [])
            logger.debug("Blockquotes identified: %s", raw_blockquotes)
            raw_code_blocks = convert_dict_keys_to_snake_case(
                analyzer.identify_code_blocks()).get("code_block", [])
            logger.debug("Code blocks identified: %s", raw_code_blocks)
            raw_lists = convert_dict_keys_to_snake_case(
                analyzer.identify_lists())
            logger.debug("Lists identified: %s", raw_lists)
            raw_tables = convert_dict_keys_to_snake_case(
                analyzer.identify_tables()).get("table", [])
            logger.debug("Tables identified: %s", raw_tables)
            raw_links = convert_dict_keys_to_snake_case(
                analyzer.identify_links())
            logger.debug("Links identified: %s", raw_links)
            raw_footnotes = convert_dict_keys_to_snake_case(
                analyzer.identify_footnotes())
            logger.debug("Footnotes identified: %s", raw_footnotes)
            raw_inline_code = convert_dict_keys_to_snake_case(
                analyzer.identify_inline_code())
            logger.debug("Inline code identified: %s", raw_inline_code)
            raw_emphasis = convert_dict_keys_to_snake_case(
                analyzer.identify_emphasis())
            logger.debug("Emphasis identified: %s", raw_emphasis)
            raw_task_items = convert_dict_keys_to_snake_case(
                analyzer.identify_task_items())
            logger.debug("Task items identified: %s", raw_task_items)
            raw_html_blocks = convert_dict_keys_to_snake_case(
                analyzer.identify_html_blocks())
            logger.debug("HTML blocks identified: %s", raw_html_blocks)
            raw_html_inline = convert_dict_keys_to_snake_case(
                analyzer.identify_html_inline())
            logger.debug("HTML inline identified: %s", raw_html_inline)
            raw_tokens_sequential = convert_dict_keys_to_snake_case(
                analyzer.get_tokens_sequential())
            logger.debug("Tokens sequential identified: %s",
                         raw_tokens_sequential)
            raw_word_count = analyzer.count_words()
            logger.debug("Word count: %s", raw_word_count)
            raw_char_count = analyzer.count_characters()
            logger.debug("Character count: %s", raw_char_count)
        except ValueError as e:
            logger.warning(
                "Error in MarkdownAnalyzer: %s. Returning partial results.", e)
            return values

        # Extract unordered_list and ordered_list from raw_lists
        unordered_lists = raw_lists.get("unordered_list", [])
        ordered_lists = raw_lists.get("ordered_list", [])

        # Extract text_link and image_link from raw_links
        text_links = raw_links.get("text_link", [])
        image_links = raw_links.get("image_link", [])
        text_link_count = len(text_links)
        image_link_count = len(image_links)
        logger.debug("Text links count: %s, Image links count: %s",
                     text_link_count, image_link_count)

        header_counts = {f"h{i}": 0 for i in range(1, 7)}
        for header in raw_headers:
            level = header.get("level")
            if level in range(1, 7):
                header_counts[f"h{level}"] += 1
            else:
                logger.warning("Invalid header level detected: %s", level)

        raw_summary_unsorted = convert_dict_keys_to_snake_case(
            analyzer.analyse())
        raw_summary_unsorted.update({
            "header_counts": header_counts,
            "text_links": text_link_count,
            "image_links": image_link_count,
        })
        # Enforce key order from DEFAULTS["summary"]
        raw_summary = {
            key: raw_summary_unsorted.get(key, DEFAULTS["summary"][key])
            for key in DEFAULTS["summary"].keys()
        }
        logger.debug("Summary: %s", raw_summary)

        result = {
            "summary": raw_summary,
            "word_count": raw_word_count,
            "char_count": raw_char_count,
            "headers": raw_headers,
            "paragraphs": raw_paragraphs,
            "blockquotes": raw_blockquotes,
            "code_blocks": raw_code_blocks,
            "unordered_lists": unordered_lists,
            "ordered_lists": ordered_lists,
            "tables": raw_tables,
            "text_links": text_links,
            "image_links": image_links,
            "footnotes": raw_footnotes,
            "inline_code": raw_inline_code,
            "emphasis": raw_emphasis,
            "task_items": raw_task_items,
            "html_blocks": raw_html_blocks,
            "html_inline": raw_html_inline,
            "tokens_sequential": raw_tokens_sequential,
        }
        values.update(result)
        logger.debug("Final result: %s", {k: v for k, v in values.items(
        ) if k != "tokens_sequential"})  # Avoid logging large tokens
        return values
    finally:
        if temp_md_path and os.path.exists(temp_md_path):
            try:
                os.remove(temp_md_path)
                logger.debug("Temporary file removed: %s", temp_md_path)
            except Exception as e:
                logger.warning(
                    f"Could not remove temporary file {temp_md_path}: %s", e)


def analyze_markdown(input: Union[str, Path], ignore_links: bool = False) -> MarkdownAnalysis:
    return base_analyze_markdown(input, ignore_links)


def get_summary(input: Union[str, Path]) -> SummaryDict:
    analysis = analyze_markdown(input, ignore_links=False)
    return analysis["summary"]


__all__ = [
    "base_analyze_markdown",
    "analyze_markdown",
    "get_summary",
]
