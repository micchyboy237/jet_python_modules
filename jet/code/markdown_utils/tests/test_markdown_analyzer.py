import pytest
from pathlib import Path
from typing import Union
from jet.code.markdown_types.markdown_analysis_types import MarkdownAnalysis, SummaryDict
from jet.code.markdown_utils._markdown_analyzer import base_analyze_markdown


@pytest.fixture
def temp_markdown_file(tmp_path: Path):
    def create_markdown(content: str) -> Path:
        file_path = tmp_path / "test.md"
        file_path.write_text(content, encoding="utf-8")
        return file_path
    return create_markdown


class TestBaseAnalyzeMarkdown:
    def test_empty_markdown(self, temp_markdown_file: callable):
        # Given: An empty markdown file
        md_file = temp_markdown_file("")
        expected_summary: SummaryDict = {
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
        }
        expected: MarkdownAnalysis = {
            "summary": expected_summary,
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

        # When: Analyzing the empty markdown
        result = base_analyze_markdown(md_file, ignore_links=False)

        # Then: The result should match the expected empty structure
        assert result == expected, "Empty markdown analysis did not match expected output"

    def test_complex_markdown(self, temp_markdown_file: callable):
        # Given: A complex markdown with various elements
        content = """# Header 1
## Header 2
This is a paragraph with **bold** text and [a link](http://example.com).

> Blockquote text

- Item 1
- [ ] Task item
- [x] Completed task

```python
print("Hello")
```

| Col1 | Col2 |
|------|------|
| Row1 | Row2 |

Inline `code` example.

[Footnote ref]: http://example.com
"""
        md_file = temp_markdown_file(content)
        expected_summary: SummaryDict = {
            "headers": 2,
            "header_counts": {"h1": 1, "h2": 1, "h3": 0, "h4": 0, "h5": 0, "h6": 0},
            "paragraphs": 1,
            "blockquotes": 1,
            "code_blocks": 1,
            "ordered_lists": 0,
            "unordered_lists": 1,
            "tables": 1,
            "html_blocks": 0,
            "html_inline_count": 0,
            "words": 12,  # Approximate, depends on analyzer
            "characters": 100,  # Approximate, depends on analyzer
            "text_links": 1,
            "image_links": 0,
        }
        expected_unordered_lists: List[List[ListItemDict]] = [
            [
                {"text": "Item 1", "task_item": False, "checked": None},
                {"text": "Task item", "task_item": True, "checked": False},
                {"text": "Completed task", "task_item": True, "checked": True},
            ]
        ]
        expected_text_links: list = [
            {"line": 3, "text": "a link", "url": "http://example.com", "alt_text": None}
        ]
        expected: MarkdownAnalysis = {
            "summary": expected_summary,
            "word_count": 12,
            "char_count": 100,
            "headers": [
                {"line": 1, "level": 1, "text": "Header 1"},
                {"line": 2, "level": 2, "text": "Header 2"},
            ],
            "paragraphs": ["This is a paragraph with **bold** text and [a link](http://example.com)."],
            "blockquotes": ["Blockquote text"],
            "code_blocks": [{"start_line": 8, "content": 'print("Hello")', "language": "python"}],
            "unordered_lists": expected_unordered_lists,
            "ordered_lists": [],
            "tables": [{"header": ["Col1", "Col2"], "rows": [["Row1", "Row2"]]}],
            "text_links": expected_text_links,
            "image_links": [],
            "footnotes": [{"line": 14, "id": "Footnote ref", "content": "http://example.com"}],
            "inline_code": [{"line": 12, "code": "code"}],
            "emphasis": [{"line": 3, "text": "bold"}],
            "task_items": [
                {"line": 6, "text": "Task item", "checked": False},
                {"line": 7, "text": "Completed task", "checked": True},
            ],
            "html_blocks": [],
            "html_inline": [],
            "tokens_sequential": [
                {"id": 1, "type": "header1", "content": "Header 1",
                    "url": None, "checked": None},
                {"id": 2, "type": "header2", "content": "Header 2",
                    "url": None, "checked": None},
                {"id": 3, "type": "paragraph",
                    "content": "This is a paragraph with **bold** text and [a link](http://example.com).", "url": None, "checked": None},
                {"id": 4, "type": "blockquote", "content": "Blockquote text",
                    "url": None, "checked": None},
                {"id": 5, "type": "unordered_list",
                    "content": "Item 1", "url": None, "checked": None},
                {"id": 6, "type": "task_item", "content": "Task item",
                    "url": None, "checked": False},
                {"id": 7, "type": "task_item", "content": "Completed task",
                    "url": None, "checked": True},
                {"id": 8, "type": "code",
                    "content": 'print("Hello")', "url": None, "checked": None},
                {"id": 9, "type": "table", "content": "| Col1 | Col2 |\n|------|------|\n| Row1 | Row2 |",
                    "url": None, "checked": None},
                {"id": 10, "type": "inline_code", "content": "code",
                    "url": None, "checked": None},
            ],
        }

        # When: Analyzing the complex markdown
        result = base_analyze_markdown(md_file, ignore_links=False)

        # Then: The result should match the expected structure
        assert result["summary"] == expected_summary, "Summary did not match"
        assert result["word_count"] == expected["word_count"], "Word count did not match"
        assert result["char_count"] == expected["char_count"], "Char count did not match"
        assert result["headers"] == expected["headers"], "Headers did not match"
        assert result["paragraphs"] == expected["paragraphs"], "Paragraphs did not match"
        assert result["blockquotes"] == expected["blockquotes"], "Blockquotes did not match"
        assert result["code_blocks"] == expected["code_blocks"], "Code blocks did not match"
        assert result["unordered_lists"] == expected["unordered_lists"], "Unordered lists did not match"
        assert result["ordered_lists"] == expected["ordered_lists"], "Ordered lists did not match"
        assert result["tables"] == expected["tables"], "Tables did not match"
        assert result["text_links"] == expected["text_links"], "Text links did not match"
        assert result["image_links"] == expected["image_links"], "Image links did not match"
        assert result["footnotes"] == expected["footnotes"], "Footnotes did not match"
        assert result["inline_code"] == expected["inline_code"], "Inline code did not match"
        assert result["emphasis"] == expected["emphasis"], "Emphasis did not match"
        assert result["task_items"] == expected["task_items"], "Task items did not match"
        assert result["html_blocks"] == expected["html_blocks"], "HTML blocks did not match"
        assert result["html_inline"] == expected["html_inline"], "HTML inline did not match"
        assert len(result["tokens_sequential"]) == len(
            expected["tokens_sequential"]), "Tokens sequential length did not match"

    def test_ignore_links(self, temp_markdown_file: callable):
        # Given: A markdown with links
        content = """[Text link](http://example.com)
![Image link](http://image.com)
"""
        md_file = temp_markdown_file(content)
        expected_summary: SummaryDict = {
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
        }
        expected: MarkdownAnalysis = {
            "summary": expected_summary,
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

        # When: Analyzing with ignore_links=True
        result = base_analyze_markdown(md_file, ignore_links=True)

        # Then: Links should be ignored
        assert result["summary"] == expected_summary, "Summary did not match for ignored links"
        assert result["text_links"] == expected["text_links"], "Text links were not ignored"
        assert result["image_links"] == expected["image_links"], "Image links were not ignored"

    def test_malformed_markdown(self, temp_markdown_file: callable):
        # Given: A malformed markdown with unclosed elements
        content = """# Header
- Item 1
  - Nested item without closing
```python
Unclosed code block
"""
        md_file = temp_markdown_file(content)
        expected_summary: SummaryDict = {
            "headers": 1,
            "header_counts": {"h1": 1, "h2": 0, "h3": 0, "h4": 0, "h5": 0, "h6": 0},
            "paragraphs": 0,
            "blockquotes": 0,
            "code_blocks": 1,
            "ordered_lists": 0,
            "unordered_lists": 1,
            "tables": 0,
            "html_blocks": 0,
            "html_inline_count": 0,
            "words": 5,
            "characters": 30,
            "text_links": 0,
            "image_links": 0,
        }
        expected_unordered_lists: List[List[ListItemDict]] = [
            [
                {"text": "Item 1", "task_item": False, "checked": None},
                {"text": "Nested item without closing",
                    "task_item": False, "checked": None},
            ]
        ]
        expected: MarkdownAnalysis = {
            "summary": expected_summary,
            "word_count": 5,
            "char_count": 30,
            "headers": [{"line": 1, "level": 1, "text": "Header"}],
            "paragraphs": [],
            "blockquotes": [],
            "code_blocks": [{"start_line": 4, "content": "Unclosed code block", "language": "python"}],
            "unordered_lists": expected_unordered_lists,
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
            "tokens_sequential": [
                {"id": 1, "type": "header1", "content": "Header",
                    "url": None, "checked": None},
                {"id": 2, "type": "unordered_list",
                    "content": "Item 1", "url": None, "checked": None},
                {"id": 3, "type": "list_item", "content": "Nested item without closing",
                    "url": None, "checked": None},
                {"id": 4, "type": "code", "content": "Unclosed code block",
                    "url": None, "checked": None},
            ],
        }

        # When: Analyzing the malformed markdown
        result = base_analyze_markdown(md_file, ignore_links=False)

        # Then: The result should handle malformed input gracefully
        assert result["summary"] == expected_summary, "Summary did not match for malformed markdown"
        assert result["headers"] == expected["headers"], "Headers did not match"
        assert result["unordered_lists"] == expected["unordered_lists"], "Unordered lists did not match"
        assert result["ordered_lists"] == expected["ordered_lists"], "Ordered lists did not match"
        assert result["code_blocks"] == expected["code_blocks"], "Code blocks did not match"
