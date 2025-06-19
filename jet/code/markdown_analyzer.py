import os
import tempfile
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, TypedDict

from mrkdwn_analysis import MarkdownAnalyzer

from jet.logger import logger

logger.set_config(level="DEBUG",
                  format='%(levelname)s - %(message)s')


class MarkdownAnalysisResult(TypedDict):
    """Typed dictionary defining the structure of markdown analysis results."""
    headers: List[Dict[str, Any]]
    paragraphs: List[Dict[str, Any]]
    blockquotes: List[Dict[str, Any]]
    code_blocks: List[Dict[str, Any]]
    lists: List[List[Dict[str, Any]]]
    tables: List[List[Dict[str, str]]]
    links: List[Dict[str, str]]
    footnotes: List[Dict[str, Any]]
    inline_code: List[str]
    emphasis: List[Dict[str, str]]
    task_items: List[str]
    html_blocks: List[str]
    html_inline: List[str]
    tokens_sequential: List[Dict[str, Any]]
    word_count: Dict[str, int]
    char_count: List[int]
    summary: Dict[str, int]


def analyze_markdown(md_input: Union[str, Path]) -> MarkdownAnalysisResult:
    temp_md_path: Optional[Path] = None
    try:
        if isinstance(md_input, (str, Path)) and str(md_input).endswith('.md'):
            if not Path(str(md_input)).is_file():
                raise OSError(f"File {md_input} does not exist")
            with open(md_input, 'r', encoding='utf-8') as file:
                md_content = file.read()
        else:
            md_content = str(md_input)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as temp_file:
            temp_file.write(md_content)
            temp_md_path = Path(temp_file.name)
        analyzer = MarkdownAnalyzer(str(temp_md_path))
        analysis_results: MarkdownAnalysisResult = {
            "headers": analyzer.identify_headers(),
            "paragraphs": analyzer.identify_paragraphs(),
            "blockquotes": analyzer.identify_blockquotes(),
            "code_blocks": analyzer.identify_code_blocks(),
            "lists": analyzer.identify_lists(),
            "tables": analyzer.identify_tables(),
            "links": analyzer.identify_links(),
            "footnotes": analyzer.identify_footnotes(),
            "inline_code": analyzer.identify_inline_code(),
            "emphasis": analyzer.identify_emphasis(),
            "task_items": analyzer.identify_task_items(),
            "html_blocks": analyzer.identify_html_blocks(),
            "html_inline": analyzer.identify_html_inline(),
            "tokens_sequential": analyzer.get_tokens_sequential(),
            "word_count": {"word_count": analyzer.count_words()},
            "char_count": [analyzer.count_characters()],
            "summary": analyzer.analyse(),
        }
        return analysis_results
    finally:
        if temp_md_path and temp_md_path.exists():
            try:
                temp_md_path.unlink()
            except OSError:
                print(
                    f"Warning: Could not delete temporary file {temp_md_path}")
