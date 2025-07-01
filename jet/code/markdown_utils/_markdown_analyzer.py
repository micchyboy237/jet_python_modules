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


# @timeout(3)
def base_analyze_markdown(input: Union[str, Path], ignore_links: bool = False) -> dict:
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
    try:
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
            return {
                "summary": {
                    "headers": 0,
                    "header_counts": {"h1": 0, "h2": 0, "h3": 0, "h4": 0, "h5": 0, "h6": 0},
                    "paragraphs": len(raw_paragraphs) if 'raw_paragraphs' in locals() else 0,
                    "blockquotes": len(raw_blockquotes) if 'raw_blockquotes' in locals() else 0,
                    "code_blocks": len(raw_code_blocks) if 'raw_code_blocks' in locals() else 0,
                    "ordered_lists": len(raw_lists.get("ordered_list", [])) if 'raw_lists' in locals() else 0,
                    "unordered_lists": len(raw_lists.get("unordered_list", [])) if 'raw_lists' in locals() else 0,
                    "tables": len(raw_tables) if 'raw_tables' in locals() else 0,
                    "html_blocks": len(raw_html_blocks) if 'raw_html_blocks' in locals() else 0,
                    "html_inline_count": len(raw_html_inline) if 'raw_html_inline' in locals() else 0,
                    "words": raw_word_count if 'raw_word_count' in locals() else 0,
                    "characters": raw_char_count if 'raw_char_count' in locals() else 0,
                    "text_links": len(raw_links.get("text_link", [])) if 'raw_links' in locals() else 0,
                    "image_links": len(raw_links.get("image_link", [])) if 'raw_links' in locals() else 0,
                },
                "word_count": raw_word_count if 'raw_word_count' in locals() else 0,
                "char_count": raw_char_count if 'raw_char_count' in locals() else 0,
                "headers": raw_headers if 'raw_headers' in locals() else [],
                "paragraphs": raw_paragraphs if 'raw_paragraphs' in locals() else [],
                "blockquotes": raw_blockquotes if 'raw_blockquotes' in locals() else [],
                "code_blocks": raw_code_blocks if 'raw_code_blocks' in locals() else [],
                "unordered_lists": raw_lists.get("unordered_list", []) if 'raw_lists' in locals() else [],
                "ordered_lists": raw_lists.get("ordered_list", []) if 'raw_lists' in locals() else [],
                "tables": raw_tables if 'raw_tables' in locals() else [],
                "text_links": raw_links.get("text_link", []) if 'raw_links' in locals() else [],
                "image_links": raw_links.get("image_link", []) if 'raw_links' in locals() else [],
                "footnotes": raw_footnotes if 'raw_footnotes' in locals() else [],
                "inline_code": raw_inline_code if 'raw_inline_code' in locals() else [],
                "emphasis": raw_emphasis if 'raw_emphasis' in locals() else [],
                "task_items": raw_task_items if 'raw_task_items' in locals() else [],
                "html_blocks": raw_html_blocks if 'raw_html_blocks' in locals() else [],
                "html_inline": raw_html_inline if 'raw_html_inline' in locals() else [],
                "tokens_sequential": raw_tokens_sequential if 'raw_tokens_sequential' in locals() else [],
            }

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

        raw_summary = convert_dict_keys_to_snake_case(analyzer.analyse())
        raw_summary = {
            **raw_summary,
            "header_counts": header_counts,
            "text_links": text_link_count,
            "image_links": image_link_count,
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
        logger.debug("Final result: %s", {k: v for k, v in result.items(
        ) if k != "tokens_sequential"})  # Avoid logging large tokens
        return result
    finally:
        if temp_md_path and os.path.exists(temp_md_path):
            try:
                os.remove(temp_md_path)
                logger.debug("Temporary file removed: %s", temp_md_path)
            except Exception as e:
                logger.warning(
                    f"Could not remove temporary file {temp_md_path}: %s", e)


def analyze_markdown(input: Union[str, Path], ignore_links: bool = False) -> MarkdownAnalysis:
    """
    Analyze markdown content and return structured analysis results.

    Args:
        input: Either a string containing markdown content or a Path to a markdown file.

    Returns:
        A dictionary containing detailed analysis of markdown elements.

    Raises:
        OSError: If the input file does not exist.
        TimeoutError: If analysis takes too long.
    """
    temp_md_path: Optional[Path] = None
    try:
        md_content = read_md_content(input, ignore_links=ignore_links)

        # Preprocess markdown
        md_content = preprocess_markdown(md_content)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as temp_file:
            temp_file.write(md_content)
            temp_md_path = Path(temp_file.name)

        (
            raw_summary, raw_headers, raw_paragraphs, raw_blockquotes, raw_code_blocks,
            raw_lists, raw_tables, raw_links, raw_footnotes, raw_inline_code,
            raw_emphasis, raw_task_items, raw_html_blocks, raw_html_inline,
            raw_tokens_sequential, word_count, char_count
        ) = analyze_with_timeout(str(temp_md_path))

        analysis_results: MarkdownAnalysis = {
            "summary": raw_summary,
            "headers": {
                "header": [
                    {**header, "text": clean_markdown_text(header["text"])}
                    for header in raw_headers.get("header", [])
                ]
            },
            "paragraphs": {
                "paragraph": [clean_markdown_text(p) for p in raw_paragraphs.get("paragraph", [])]
            },
            "blockquotes": {
                "blockquote": [clean_markdown_text(b) for b in raw_blockquotes.get("blockquote", [])]
            },
            "code_blocks": {
                "code_block": [
                    {**cb, "content": clean_markdown_text(cb["content"])}
                    for cb in raw_code_blocks.get("code_block", [])
                ]
            },
            "lists": {
                key: [
                    [{**item, "text": clean_markdown_text(item["text"])}
                     for item in sublist]
                    for sublist in v
                ]
                for key, v in raw_lists.items()
            },
            "tables": {
                "table": [
                    {
                        "header": [clean_markdown_text(h) for h in table["header"]],
                        "rows": [
                            [clean_markdown_text(cell) for cell in row]
                            for row in table["rows"]
                        ]
                    }
                    for table in raw_tables.get("table", [])
                ]
            },
            "links": {
                key: [
                    {
                        **link,
                        "text": clean_markdown_text(link.get("text")) if link.get("text") is not None else None,
                        "alt_text": clean_markdown_text(link.get("alt_text")) if link.get("alt_text") is not None else None,
                        "line": link["line"],
                        "url": link["url"]
                    }
                    for link in v
                ]
                for key, v in raw_links.items()
            },
            "footnotes": [
                {**fn, "content": clean_markdown_text(fn["content"])}
                for fn in raw_footnotes
            ],
            "inline_code": [
                {**ic, "code": clean_markdown_text(ic["code"])}
                for ic in raw_inline_code
            ],
            "emphasis": [
                {
                    "emphasis": {**em, "text": clean_markdown_text(em["text"])},
                    "text": clean_markdown_text(em["text"])
                }
                for em in raw_emphasis
            ],
            "task_items": [
                {**ti, "text": clean_markdown_text(ti["text"])}
                for ti in raw_task_items
            ],
            "html_blocks": [
                {**hb, "content": clean_markdown_text(hb["content"])}
                for hb in raw_html_blocks
            ],
            "html_inline": [
                {**hi, "html": clean_markdown_text(hi["html"])}
                for hi in raw_html_inline
            ],
            "tokens_sequential": [
                {**token, "content": clean_markdown_text(token["content"])}
                for token in raw_tokens_sequential
            ],
            "word_count": {"word_count": word_count},
            "char_count": {"char": char_count}
        }

        return analysis_results

    except TimeoutError as e:
        logger.error(f"Analysis timed out: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error analyzing markdown: {str(e)}")
        raise
    finally:
        if temp_md_path and temp_md_path.exists():
            try:
                temp_md_path.unlink()
            except OSError:
                logger.warning(
                    f"Warning: Could not delete temporary file {temp_md_path}")


def get_summary(input: Union[str, Path]) -> SummaryDict:
    temp_md_path: Optional[Path] = None
    try:
        md_content = read_md_content(input)
        md_content = preprocess_markdown(md_content)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as temp_file:
            temp_file.write(md_content)
            temp_md_path = Path(temp_file.name)
        analyzer = MarkdownAnalyzer(str(temp_md_path))

        # Get headers for counting individual levels
        headers = convert_dict_keys_to_snake_case(
            analyzer.identify_headers()).get("header", [])
        header_counts = {f"h{i}": 0 for i in range(1, 7)}
        for header in headers:
            level = header.get("level")
            if level in range(1, 7):
                header_counts[f"h{level}"] += 1
            else:
                logger.warning("Invalid header level detected: %s", level)

        # Get links for counting
        links = convert_dict_keys_to_snake_case(analyzer.identify_links())
        text_link_count = len(links.get("text_link", []))
        image_link_count = len(links.get("image_link", []))

        # Get existing analysis
        raw_summary = convert_dict_keys_to_snake_case(analyzer.analyse())

        # Update summary with header counts and link counts
        summary = {
            **raw_summary,
            "header_counts": header_counts,
            "text_links": text_link_count,
            "image_links": image_link_count,
        }

        return summary
    finally:
        if temp_md_path and temp_md_path.exists():
            try:
                temp_md_path.unlink()
            except OSError:
                logger.warning(
                    f"Warning: Could not delete temporary file {temp_md_path}")


__all__ = [
    "base_analyze_markdown",
    "analyze_markdown",
    "get_summary",
]
