import os
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, TypedDict

from mrkdwn_analysis import MarkdownAnalyzer


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
    analysis: Dict[str, Any]


def analyze_markdown(md_input: Union[str, Path]) -> MarkdownAnalysisResult:
    """
    Analyzes markdown content from either a string or a file path.

    Args:
        md_input: Markdown content as a string or path to a markdown file.

    Returns:
        Typed dictionary containing all analysis results.

    Raises:
        OSError: If file operations fail or the specified file does not exist.
        Exception: For other unexpected errors during analysis.
    """
    temp_md_path: Optional[Path] = None
    try:
        # Handle input: read file if path, otherwise use string content
        if isinstance(md_input, (str, Path)) and str(md_input).endswith('.md'):
            if not Path(str(md_input)).is_file():
                raise OSError(f"File {md_input} does not exist")
            with open(md_input, 'r', encoding='utf-8') as file:
                md_content = file.read()
        else:
            md_content = str(md_input)

        # Create temporary markdown file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as temp_file:
            temp_file.write(md_content)
            temp_md_path = Path(temp_file.name)

        # Initialize analyzer
        analyzer = MarkdownAnalyzer(str(temp_md_path))

        # Get raw lists and process to separate ordered lists
        raw_lists = analyzer.identify_lists()
        processed_lists = []
        for lst in raw_lists:
            unordered = []
            ordered = []
            for item in lst:
                if item.get('text', '').startswith(('1. ', '2. ')):
                    ordered.append({"text": item['text'].split('\n')[
                                   0], "type": "ordered", "task_item": item.get('task_item', False)})
                else:
                    unordered.append({**item, "type": "unordered"})
            if unordered:
                processed_lists.append(unordered)
            if ordered:
                processed_lists.append(ordered)

        # Get raw tokens and populate list content
        tokens = analyzer.get_tokens_sequential()
        updated_tokens = []
        i = 0
        while i < len(tokens):
            token = tokens[i]
            updated_tokens.append(token)
            if token.get('type') == 'unordered_list':
                # Collect content from subsequent list_item or task_item tokens
                content_items = []
                j = i + 1
                while j < len(tokens) and tokens[j].get('type') in ('list_item', 'task_item'):
                    content_items.append(tokens[j].get('content', ''))
                    j += 1
                token['content'] = '\n'.join(content_items)
            elif token.get('type') == 'list_item' and token.get('content', '').startswith(('1. ', '2. ')):
                # Insert an ordered_list token before ordered list items
                content_items = []
                j = i
                while j < len(tokens) and tokens[j].get('type') == 'list_item' and tokens[j].get('content', '').startswith(('1. ', '2. ')):
                    content_items.append(tokens[j].get(
                        'content', '').split('\n')[0])
                    j += 1
                if content_items:
                    updated_tokens.insert(len(updated_tokens) - 1, {
                        'type': 'ordered_list',
                        'content': '\n'.join(content_items)
                    })
            i += 1

        # Perform analysis
        analysis_results: MarkdownAnalysisResult = {
            "headers": analyzer.identify_headers(),
            "paragraphs": analyzer.identify_paragraphs(),
            "blockquotes": analyzer.identify_blockquotes(),
            "code_blocks": analyzer.identify_code_blocks(),
            "lists": processed_lists,
            "tables": analyzer.identify_tables(),
            "links": analyzer.identify_links(),
            "footnotes": analyzer.identify_footnotes(),
            "inline_code": analyzer.identify_inline_code(),
            "emphasis": analyzer.identify_emphasis(),
            "task_items": analyzer.identify_task_items(),
            "html_blocks": analyzer.identify_html_blocks(),
            "html_inline": analyzer.identify_html_inline(),
            "tokens_sequential": updated_tokens,
            "word_count": {"word_count": analyzer.count_words()},
            "char_count": [analyzer.count_characters()],
            "analysis": analyzer.analyse(),
        }

        return analysis_results

    finally:
        # Clean up temporary file
        if temp_md_path and temp_md_path.exists():
            try:
                temp_md_path.unlink()
            except OSError:
                print(
                    f"Warning: Could not delete temporary file {temp_md_path}")
