import tempfile
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, TypedDict, cast

import html2text
from jet.code.html_utils import preprocess_html, valid_html
from jet.code.markdown_types import MarkdownAnalysis, MarkdownToken, SummaryDict
from jet.code.markdown_types.base_markdown_analysis_types import (
    BaseMarkdownAnalysis,
    HeaderCounts,
    Header,
    CodeBlock,
    Table,
    ListItem,
    TextLink,
    ImageLink,
    Footnote,
    InlineCode,
    Emphasis,
    TaskItem,
    HtmlBlock,
    TokenSequential,
    Analysis,
    BaseMarkdownAnalysis,
)
from jet.code.markdown_utils import read_md_content, preprocess_markdown
from jet.code.markdown_utils._preprocessors import clean_markdown_links
from jet.decorators.timer import timeout
from jet.transformers.object import convert_dict_keys_to_snake_case, make_serializable
from jet.utils.text import fix_and_unidecode
from mrkdwn_analysis import MarkdownAnalyzer, MarkdownParser

from jet.logger import logger


BASE_DEFAULTS: BaseMarkdownAnalysis = {
    "analysis": Analysis(
        headers=0,
        paragraphs=0,
        blockquotes=0,
        code_blocks=0,
        ordered_lists=0,
        unordered_lists=0,
        tables=0,
        html_blocks=0,
        html_inline_count=0,
        words=0,
        characters=0,
        header_counts=HeaderCounts(h1=0, h2=0, h3=0, h4=0, h5=0, h6=0),
        text_links=0,
        image_links=0
    ),
    "header": [],
    "paragraph": [],
    "blockquote": [],
    "code_block": [],
    "table": [],
    "unordered_list": [],
    "ordered_list": [],
    "text_link": [],
    "image_link": [],
    "footnotes": [],
    "inline_code": [],
    "emphasis": [],
    "task_items": [],
    "html_inline": [],
    "html_blocks": [],
    "tokens_sequential": []  # Empty list is still valid with updated TokenSequential
}


# @timeout(3)

def validate_analysis(data: Dict[str, Any]) -> Analysis:
    expected_keys = set(Analysis.__annotations__.keys())
    if not all(key in expected_keys for key in data):
        logger.warning("Unexpected keys in analysis: %s",
                       set(data.keys()) - expected_keys)
    if not isinstance(data.get("header_counts", {}), dict) or not all(k in HeaderCounts.__annotations__ for k in data.get("header_counts", {})):
        logger.warning("Invalid header_counts structure")
        data["header_counts"] = BASE_DEFAULTS["analysis"]["header_counts"]
    return cast(Analysis, {k: data.get(k, BASE_DEFAULTS["analysis"][k]) for k in expected_keys})


def base_analyze_markdown(input: Union[str, Path], ignore_links: bool = False) -> BaseMarkdownAnalysis:
    md_content = read_md_content(input, ignore_links=ignore_links)
    md_content = preprocess_markdown(md_content)
    temp_md_path = None
    values: BaseMarkdownAnalysis = BASE_DEFAULTS.copy()
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".md", mode="w", encoding="utf-8") as tmpfile:
            tmpfile.write(md_content)
            temp_md_path = tmpfile.name
        logger.debug("Temporary markdown file created at: %s", temp_md_path)
        analyzer = MarkdownAnalyzer(temp_md_path)

        # Extract data from analyzer
        try:
            raw_headers = convert_dict_keys_to_snake_case(
                analyzer.identify_headers()).get("header", [])
            raw_links = convert_dict_keys_to_snake_case(
                analyzer.identify_links())
            text_links = raw_links.get("text_link", [])
            image_links = raw_links.get("image_link", [])

            # Calculate header_counts
            header_counts: HeaderCounts = {f"h{i}": 0 for i in range(1, 7)}
            for header in raw_headers:
                level = header.get("level")
                if level in range(1, 7):
                    header_counts[f"h{level}"] += 1
                else:
                    logger.warning("Invalid header level detected: %s", level)

            # Update analysis with new fields
            raw_analysis = convert_dict_keys_to_snake_case(analyzer.analyse())
            raw_analysis.update({
                "header_counts": header_counts,
                "text_links": len(text_links),
                "image_links": len(image_links),
            })

            # Update values with analyzer results
            values.update({
                "analysis": validate_analysis(raw_analysis),
                "header": cast(List[Header], raw_headers),
                "paragraph": cast(List[str], convert_dict_keys_to_snake_case(analyzer.identify_paragraphs()).get("paragraph", [])),
                "blockquote": cast(List[str], convert_dict_keys_to_snake_case(analyzer.identify_blockquotes()).get("blockquote", [])),
                "code_block": cast(List[CodeBlock], convert_dict_keys_to_snake_case(analyzer.identify_code_blocks()).get("code_block", [])),
                "table": cast(List[Table], convert_dict_keys_to_snake_case(analyzer.identify_tables()).get("table", [])),
                "unordered_list": cast(List[List[ListItem]], convert_dict_keys_to_snake_case(analyzer.identify_lists()).get("unordered_list", [])),
                "ordered_list": cast(List[List[ListItem]], convert_dict_keys_to_snake_case(analyzer.identify_lists()).get("ordered_list", [])),
                "text_link": cast(List[TextLink], text_links),
                "image_link": cast(List[ImageLink], image_links),
                "footnotes": cast(List[Footnote], analyzer.identify_footnotes()),
                "inline_code": cast(List[InlineCode], analyzer.identify_inline_code()),
                "emphasis": cast(List[Emphasis], analyzer.identify_emphasis()),
                "task_items": cast(List[TaskItem], analyzer.identify_task_items()),
                "html_inline": cast(List[str], analyzer.identify_html_inline()),
                "html_blocks": cast(List[HtmlBlock], analyzer.identify_html_blocks()),
                "tokens_sequential": cast(List[TokenSequential], analyzer.get_tokens_sequential()),
            })
        except ValueError as e:
            logger.warning(
                "Error in MarkdownAnalyzer: %s. Returning default values.", e)
            return values

        logger.debug("Final result: %s", {
                     k: v for k, v in values.items() if k != "tokens_sequential"})
        return values
    finally:
        if temp_md_path and os.path.exists(temp_md_path):
            try:
                os.remove(temp_md_path)
                logger.debug("Temporary file removed: %s", temp_md_path)
            except Exception as e:
                logger.warning(
                    "Could not remove temporary file %s: %s", temp_md_path, e)


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


def analyze_markdown(input: Union[str, Path], ignore_links: bool = False) -> MarkdownAnalysis:
    md_content = read_md_content(input, ignore_links=ignore_links)
    temp_md_path = None
    values: MarkdownAnalysis = DEFAULTS.copy()
    try:
        md_content = preprocess_markdown(md_content)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".md", mode="w", encoding="utf-8") as tmpfile:
            tmpfile.write(md_content)
            temp_md_path = tmpfile.name
        analyzer = MarkdownAnalyzer(temp_md_path)

        # Flatten dictionaries by extracting nested lists
        try:
            raw_headers = convert_dict_keys_to_snake_case(
                analyzer.identify_headers()).get("header", [])
            raw_paragraphs = convert_dict_keys_to_snake_case(
                analyzer.identify_paragraphs()).get("paragraph", [])
            raw_blockquotes = convert_dict_keys_to_snake_case(
                analyzer.identify_blockquotes()).get("blockquote", [])
            raw_code_blocks = convert_dict_keys_to_snake_case(
                analyzer.identify_code_blocks()).get("code_block", [])
            raw_lists = convert_dict_keys_to_snake_case(
                analyzer.identify_lists())
            raw_tables = convert_dict_keys_to_snake_case(
                analyzer.identify_tables()).get("table", [])
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
            raw_word_count = analyzer.count_words()
            raw_char_count = analyzer.count_characters()
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
        return values
    finally:
        if temp_md_path and os.path.exists(temp_md_path):
            try:
                os.remove(temp_md_path)
            except Exception as e:
                logger.warning(
                    f"Could not remove temporary file {temp_md_path}: %s", e)


def get_summary(input: Union[str, Path]) -> SummaryDict:
    analysis = analyze_markdown(input, ignore_links=False)
    return analysis["summary"]


__all__ = [
    "base_analyze_markdown",
    "analyze_markdown",
    "get_summary",
]
