import os
from pathlib import Path

import pytest
from jet.code.markdown_types import MarkdownAnalysis, MarkdownToken, SummaryDict
from jet.code.markdown_utils import (
    analyze_markdown,
    clean_markdown_text,
    convert_html_to_markdown,
    parse_markdown,
    summarize_markdown,
)


class TestCleanMarkdownText:
    def test_removes_escaped_period(self):
        # Given: A string with an escaped period
        input_text = "10\\. The Water Magician"
        expected = "10. The Water Magician"

        # When: Cleaning the text
        result = clean_markdown_text(input_text)

        # Then: The escaped period should be removed
        assert result == expected

    def test_handles_none_input(self):
        # Given: A None input
        input_text: str | None = None
        expected: str | None = None

        # When: Cleaning the text
        result = clean_markdown_text(input_text)

        # Then: The result should be None
        assert result == expected

    def test_no_escapes(self):
        # Given: A string without escaped characters
        input_text = "Hello World"
        expected = "Hello World"

        # When: Cleaning the text
        result = clean_markdown_text(input_text)

        # Then: The text should remain unchanged
        assert result == expected

    def test_multiple_escaped_periods(self):
        # Given: A string with multiple escaped periods
        input_text = "1\\.2\\.3\\. Version"
        expected = "1.2.3. Version"

        # When: Cleaning the text
        result = clean_markdown_text(input_text)

        # Then: All escaped periods should be removed
        assert result == expected


class TestConvertHtmlToMarkdown:
    def test_converts_basic_html(self):
        # Given: A simple HTML string
        html_input = "<h1>Hello</h1><p>World</p>"
        expected = "# Hello\n\nWorld"

        # When: Converting HTML to markdown
        result = convert_html_to_markdown(html_input)

        # Then: The markdown should match the expected output
        assert result.strip() == expected.strip()

    def test_converts_html_file(self, tmp_path: Path):
        # Given: An HTML file
        html_content = "<h2>Test</h2><p>Content</p>"
        expected = "## Test\n\nContent"
        html_file = tmp_path / "test.html"
        html_file.write_text(html_content, encoding="utf-8")

        # When: Converting the HTML file to markdown
        result = convert_html_to_markdown(html_file)

        # Then: The markdown should match the expected output
        assert result.strip() == expected.strip()

    def test_raises_error_for_nonexistent_file(self, tmp_path: Path):
        # Given: A nonexistent HTML file
        html_file = tmp_path / "nonexistent.html"
        expected_error = f"File {html_file} does not exist"

        # When: Attempting to convert the nonexistent file
        with pytest.raises(OSError) as exc_info:
            result = convert_html_to_markdown(html_file)

        # Then: An OSError should be raised with the expected message
        assert str(exc_info.value) == expected_error


class TestParseMarkdown:
    def test_parses_header_with_escaped_period(self, tmp_path: Path):
        # Given: A markdown file with a header containing an escaped period
        markdown_content = "### 10\\. The Water Magician"
        expected: list[MarkdownToken] = [
            {
                "type": "header",
                "content": "### 10. The Water Magician",
                "level": 3,
                "meta": {},
                "line": 1,
            }
        ]
        md_file = tmp_path / "test.md"
        md_file.write_text(markdown_content, encoding="utf-8")

        # When: Parsing the markdown file
        result = parse_markdown(md_file)

        # Then: The parsed token should have cleaned content
        assert result == expected

    def test_parses_string_input(self):
        # Given: A markdown string with a paragraph
        markdown_content = "Simple text"
        expected: list[MarkdownToken] = [
            {
                "type": "paragraph",
                "content": "Simple text",
                "level": None,
                "meta": {},
                "line": 1,
            }
        ]

        # When: Parsing the markdown string
        result = parse_markdown(markdown_content)

        # Then: The parsed token should match the expected output
        assert result == expected

    def test_raises_error_for_nonexistent_file(self, tmp_path: Path):
        # Given: A nonexistent markdown file
        md_file = tmp_path / "nonexistent.md"
        expected_error = f"File {md_file} does not exist"

        # When: Attempting to parse the nonexistent file
        with pytest.raises(OSError) as exc_info:
            result = parse_markdown(md_file)

        # Then: An OSError should be raised with the expected message
        assert str(exc_info.value) == expected_error

    def test_handles_empty_markdown(self):
        # Given: An empty markdown string
        markdown_content = ""
        expected: list[MarkdownToken] = []

        # When: Parsing the empty markdown
        result = parse_markdown(markdown_content)

        # Then: An empty list of tokens should be returned
        assert result == expected


class TestAnalyzeMarkdown:
    def test_analyzes_header_with_escaped_period(self, tmp_path: Path):
        # Given: A markdown file with a header containing an escaped period
        markdown_content = "### 10\\. The Water Magician"
        expected: MarkdownAnalysis = {
            "headers": {
                "Header": [{"line": 1, "level": 3, "text": "10. The Water Magician"}]
            },
            "paragraphs": {"Paragraph": []},
            "blockquotes": {"Blockquote": []},
            "code_blocks": {"Code block": []},
            "lists": {"Ordered list": [], "Unordered list": []},
            "tables": {"Table": []},
            "links": {"Text link": [], "Image link": []},
            "footnotes": [],
            "inline_code": [],
            "emphasis": [],
            "task_items": [],
            "html_blocks": [],
            "html_inline": [],
            "tokens_sequential": [
                {
                    "type": "header",
                    "content": "10. The Water Magician",
                    "level": 3,
                    "meta": {},
                    "line": 1,
                }
            ],
            "word_count": {"word_count": 3},
            "char_count": [18],
            "summary": {
                "headers": 1,
                "paragraphs": 0,
                "blockquotes": 0,
                "code_blocks": 0,
                "ordered_lists": 0,
                "unordered_lists": 0,
                "tables": 0,
                "html_blocks": 0,
                "html_inline_count": 0,
                "words": 3,
                "characters": 18,
            },
        }
        md_file = tmp_path / "test.md"
        md_file.write_text(markdown_content, encoding="utf-8")

        # When: Analyzing the markdown file
        result = analyze_markdown(md_file)

        # Then: The analysis should have cleaned header text
        assert result == expected

    def test_analyzes_code_block_with_escaped_period(self, tmp_path: Path):
        # Given: A markdown file with a code block containing an escaped period
        markdown_content = "```python\nprint('10\\. The Water Magician')\n```"
        expected: MarkdownAnalysis = {
            "headers": {"Header": []},
            "paragraphs": {"Paragraph": []},
            "blockquotes": {"Blockquote": []},
            "code_blocks": {
                "Code block": [
                    {
                        "start_line": 1,
                        "content": "print('10. The Water Magician')",
                        "language": "python",
                    }
                ]
            },
            "lists": {"Ordered list": [], "Unordered list": []},
            "tables": {"Table": []},
            "links": {"Text link": [], "Image link": []},
            "footnotes": [],
            "inline_code": [],
            "emphasis": [],
            "task_items": [],
            "html_blocks": [],
            "html_inline": [],
            "tokens_sequential": [
                {
                    "type": "code",
                    "content": "print('10. The Water Magician')",
                    "level": None,
                    "meta": {"language": "python", "code_type": "indented"},
                    "line": 1,
                }
            ],
            "word_count": {"word_count": 4},
            "char_count": [31],
            "summary": {
                "headers": 0,
                "paragraphs": 0,
                "blockquotes": 0,
                "code_blocks": 1,
                "ordered_lists": 0,
                "unordered_lists": 0,
                "tables": 0,
                "html_blocks": 0,
                "html_inline_count": 0,
                "words": 4,
                "characters": 31,
            },
        }
        md_file = tmp_path / "test.md"
        md_file.write_text(markdown_content, encoding="utf-8")

        # When: Analyzing the markdown file
        result = analyze_markdown(md_file)

        # Then: The analysis should have cleaned code block content
        assert result == expected

    def test_analyzes_list_with_escaped_period(self, tmp_path: Path):
        # Given: A markdown file with a list containing an escaped period
        markdown_content = "- Item 1\\.0"
        expected: MarkdownAnalysis = {
            "headers": {"Header": []},
            "paragraphs": {"Paragraph": []},
            "blockquotes": {"Blockquote": []},
            "code_blocks": {"Code block": []},
            "lists": {
                "Ordered list": [],
                "Unordered list": [
                    [{"text": "Item 1.0", "task_item": False, "checked": False}]
                ],
            },
            "tables": {"Table": []},
            "links": {"Text link": [], "Image link": []},
            "footnotes": [],
            "inline_code": [],
            "emphasis": [],
            "task_items": [],
            "html_blocks": [],
            "html_inline": [],
            "tokens_sequential": [
                {
                    "type": "unordered_list",
                    "content": "Item 1.0",
                    "level": 1,
                    "meta": {"text": "Item 1.0", "task_item": False, "checked": False},
                    "line": 1,
                }
            ],
            "word_count": {"word_count": 2},
            "char_count": [10],
            "summary": {
                "headers": 0,
                "paragraphs": 0,
                "blockquotes": 0,
                "code_blocks": 0,
                "ordered_lists": 0,
                "unordered_lists": 1,
                "tables": 0,
                "html_blocks": 0,
                "html_inline_count": 0,
                "words": 2,
                "characters": 10,
            },
        }
        md_file = tmp_path / "test.md"
        md_file.write_text(markdown_content, encoding="utf-8")

        # When: Analyzing the markdown file
        result = analyze_markdown(md_file)

        # Then: The analysis should have cleaned list item text
        assert result == expected

    def test_raises_error_for_nonexistent_file(self, tmp_path: Path):
        # Given: A nonexistent markdown file
        md_file = tmp_path / "nonexistent.md"
        expected_error = f"File {md_file} does not exist"

        # When: Attempting to analyze the nonexistent file
        with pytest.raises(OSError) as exc_info:
            result = analyze_markdown(md_file)

        # Then: An OSError should be raised with the expected message
        assert str(exc_info.value) == expected_error

    def test_handles_empty_markdown(self, tmp_path: Path):
        # Given: An empty markdown file
        markdown_content = ""
        expected: MarkdownAnalysis = {
            "headers": {"Header": []},
            "paragraphs": {"Paragraph": []},
            "blockquotes": {"Blockquote": []},
            "code_blocks": {"Code block": []},
            "lists": {"Ordered list": [], "Unordered list": []},
            "tables": {"Table": []},
            "links": {"Text link": [], "Image link": []},
            "footnotes": [],
            "inline_code": [],
            "emphasis": [],
            "task_items": [],
            "html_blocks": [],
            "html_inline": [],
            "tokens_sequential": [],
            "word_count": {"word_count": 0},
            "char_count": [0],
            "summary": {
                "headers": 0,
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
            },
        }
        md_file = tmp_path / "test.md"
        md_file.write_text(markdown_content, encoding="utf-8")

        # When: Analyzing the empty markdown file
        result = analyze_markdown(md_file)

        # Then: The analysis should return empty results
        assert result == expected

    def test_temporary_file_cleanup(self, tmp_path: Path):
        # Given: A markdown file to analyze
        markdown_content = "# Test"
        md_file = tmp_path / "test.md"
        md_file.write_text(markdown_content, encoding="utf-8")
        temp_dir = tmp_path / "temp"
        temp_dir.mkdir()
        original_tempdir = os.environ.get("TMPDIR")
        os.environ["TMPDIR"] = str(temp_dir)

        # When: Analyzing the markdown file
        result = analyze_markdown(md_file)

        # Then: No temporary .md files should remain
        temp_files = list(temp_dir.glob("*.md"))
        assert len(temp_files) == 0

        # Cleanup environment
        if original_tempdir:
            os.environ["TMPDIR"] = original_tempdir
        else:
            del os.environ["TMPDIR"]


class TestGetSummary:
    def test_header_counts(self, tmp_path: Path):
        # Given: A markdown file with various header levels
        md_content = """
        # Header 1
        ## Header 2
        ### Header 3
        # Another Header 1
        ## Another Header 2
        """
        md_file = tmp_path / "test.md"
        md_file.write_text(md_content, encoding="utf-8")

        # When: Analyzing the markdown content
        result: SummaryDict = summarize_markdown(md_file)

        # Then: Verify header counts
        expected = {
            "h1": 2,
            "h2": 2,
            "h3": 1,
            "h4": 0,
            "h5": 0,
            "h6": 0,
        }
        for key, value in expected.items():
            assert result["header_counts"][key] == value, (
                f"Expected header_counts[{key}] to be {value}, but got {result['header_counts'][key]}"
            )

    def test_empty_markdown(self, tmp_path: Path):
        # Given: An empty markdown file
        md_file = tmp_path / "test.md"
        md_file.write_text("", encoding="utf-8")

        # When: Analyzing the empty markdown
        result: SummaryDict = summarize_markdown(md_file)

        # Then: Verify all header counts are zero
        expected = {
            "h1": 0,
            "h2": 0,
            "h3": 0,
            "h4": 0,
            "h5": 0,
            "h6": 0,
        }
        for key, value in expected.items():
            assert result["header_counts"][key] == value, (
                f"Expected header_counts[{key}] to be {value}, but got {result['header_counts'][key]}"
            )
        assert result["headers"] == 0, (
            f"Expected headers to be 0, but got {result['headers']}"
        )

    def test_mixed_content(self, tmp_path: Path):
        # Given: A markdown file with headers, paragraphs, and lists
        md_content = """
        # Header 1
        Some paragraph text.
        ## Header 2
        - List item 1
        - List item 2
        ### Header 3
        """
        md_file = tmp_path / "test.md"
        md_file.write_text(md_content, encoding="utf-8")

        # When: Analyzing the markdown content
        result: SummaryDict = summarize_markdown(md_file)

        # Then: Verify header counts and other summary fields
        expected = {
            "h1": 1,
            "h2": 1,
            "h3": 1,
            "h4": 0,
            "h5": 0,
            "h6": 0,
        }
        for key, value in expected.items():
            assert result["header_counts"][key] == value, (
                f"Expected header_counts[{key}] to be {value}, but got {result['header_counts'][key]}"
            )
        assert result["headers"] == 3, (
            f"Expected headers to be 3, but got {result['headers']}"
        )
        assert result["paragraphs"] == 1, (
            f"Expected paragraphs to be 1, but got {result['paragraphs']}"
        )
        assert result["unordered_lists"] == 1, (
            f"Expected unordered_lists to be 1, but got {result['unordered_lists']}"
        )
