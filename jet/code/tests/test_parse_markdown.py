from typing import List, Union, TypedDict, Literal
from pathlib import Path
import pytest
from unittest.mock import patch, mock_open
from jet.code.markdown_utils import parse_markdown, read_md_content


class TestParseMarkdownMergeParagraphs:
    """Test suite for parse_markdown function using BDD principles."""

    def test_parse_single_paragraph(self):
        """Test parsing a single paragraph."""
        # Given
        input_md = "This is a single paragraph."
        expected = [
            {
                "type": "paragraph",
                "content": "This is a single paragraph.",
                "level": None,
                "meta": {},
                "line": 1
            }
        ]

        # When
        with patch("jet.code.markdown_utils.read_md_content", return_value=input_md):
            with patch("jet.code.markdown_utils.MarkdownParser.parse", return_value=[
                {"type": "paragraph", "content": "This is a single paragraph.",
                    "level": None, "meta": {}, "line": 1}
            ]):
                result = parse_markdown("dummy_input", merge_paragraphs=True)

        # Then
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_merge_consecutive_paragraphs(self):
        """Test merging consecutive paragraphs into a single token."""
        # Given
        input_md = "First paragraph.\nSecond paragraph."
        expected = [
            {
                "type": "paragraph",
                "content": "First paragraph.\nSecond paragraph.",
                "level": None,
                "meta": {},
                "line": 1
            }
        ]

        # When
        with patch("jet.code.markdown_utils.read_md_content", return_value=input_md):
            with patch("jet.code.markdown_utils.MarkdownParser.parse", return_value=[
                {"type": "paragraph", "content": "First paragraph.",
                    "level": None, "meta": {}, "line": 1},
                {"type": "paragraph", "content": "Second paragraph.",
                    "level": None, "meta": {}, "line": 2}
            ]):
                result = parse_markdown("dummy_input", merge_paragraphs=True)

        # Then
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_paragraphs_with_non_paragraph_token(self):
        """Test merging paragraphs followed by a non-paragraph token."""
        # Given
        input_md = "Para 1.\nPara 2.\n# Header"
        expected = [
            {
                "type": "paragraph",
                "content": "Para 1.\nPara 2.",
                "level": None,
                "meta": {},
                "line": 1
            },
            {
                "type": "header",
                "content": "# Header",
                "level": 1,
                "meta": {},
                "line": 3
            }
        ]

        # When
        with patch("jet.code.markdown_utils.read_md_content", return_value=input_md):
            with patch("jet.code.markdown_utils.MarkdownParser.parse", return_value=[
                {"type": "paragraph", "content": "Para 1.",
                    "level": None, "meta": {}, "line": 1},
                {"type": "paragraph", "content": "Para 2.",
                    "level": None, "meta": {}, "line": 2},
                {"type": "header", "content": "Header",
                    "level": 1, "meta": {}, "line": 3}
            ]):
                result = parse_markdown("dummy_input", merge_paragraphs=True)

        # Then
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_non_paragraph_tokens_only(self):
        """Test handling of non-paragraph tokens without merging."""
        # Given
        input_md = "# Header\n```code\nprint('hello')\n```"
        expected = [
            {
                "type": "header",
                "content": "# Header",
                "level": 1,
                "meta": {},
                "line": 1
            },
            {
                "type": "code",
                "content": "print('hello')",
                "level": None,
                "meta": {"language": None, "code_type": "indented"},
                "line": 2
            }
        ]

        # When
        with patch("jet.code.markdown_utils.read_md_content", return_value=input_md):
            with patch("jet.code.markdown_utils.MarkdownParser.parse", return_value=[
                {"type": "header", "content": "Header",
                    "level": 1, "meta": {}, "line": 1},
                {"type": "code", "content": "print('hello')", "level": None, "meta": {
                    "language": None, "code_type": "indented"}, "line": 2}
            ]):
                result = parse_markdown("dummy_input", merge_paragraphs=True)

        # Then
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_empty_input(self):
        """Test handling of empty markdown input."""
        # Given
        input_md = ""
        expected = []

        # When
        with patch("jet.code.markdown_utils.read_md_content", return_value=input_md):
            with patch("jet.code.markdown_utils.MarkdownParser.parse", return_value=[]):
                result = parse_markdown("dummy_input", merge_paragraphs=True)

        # Then
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_html_input_conversion(self):
        """Test parsing markdown from HTML input."""
        # Given
        input_html = "<p>Paragraph 1</p><p>Paragraph 2</p><h1>Header</h1>"
        expected = [
            {
                "type": "paragraph",
                "content": "Paragraph 1\nParagraph 2",
                "level": None,
                "meta": {},
                "line": 1
            },
            {
                "type": "header",
                "content": "# Header",
                "level": 1,
                "meta": {},
                "line": 3
            }
        ]

        # When
        with patch("jet.code.markdown_utils.read_md_content", return_value="Paragraph 1\nParagraph 2\n# Header"):
            with patch("jet.code.markdown_utils.MarkdownParser.parse", return_value=[
                {"type": "paragraph", "content": "Paragraph 1",
                    "level": None, "meta": {}, "line": 1},
                {"type": "paragraph", "content": "Paragraph 2",
                    "level": None, "meta": {}, "line": 2},
                {"type": "header", "content": "Header",
                    "level": 1, "meta": {}, "line": 3}
            ]):
                result = parse_markdown("dummy.html", merge_paragraphs=True)

        # Then
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_remove_placeholder_headers(self):
        """Test removal of placeholder header tokens."""
        # Given
        input_md = "placeholder\nNormal content"
        expected = [
            {
                "type": "paragraph",
                "content": "Normal content",
                "level": None,
                "meta": {},
                "line": 2
            }
        ]

        # When
        with patch("jet.code.markdown_utils.read_md_content", return_value=input_md):
            with patch("jet.code.markdown_utils.MarkdownParser.parse", return_value=[
                {"type": "header", "content": "placeholder",
                    "level": 1, "meta": {}, "line": 1},
                {"type": "paragraph", "content": "Normal content",
                    "level": None, "meta": {}, "line": 2}
            ]):
                result = parse_markdown("dummy_input", merge_paragraphs=True)

        # Then
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_timeout_error_handling(self):
        """Test handling of TimeoutError during parsing."""
        # Given
        input_md = "Some content"

        # When
        with patch("jet.code.markdown_utils.read_md_content", return_value=input_md):
            with patch("jet.code.markdown_utils.MarkdownParser.parse", side_effect=TimeoutError("Parsing timed out")):
                # Then
                with pytest.raises(TimeoutError, match="Parsing timed out"):
                    parse_markdown("dummy_input", merge_paragraphs=True)

    def test_general_exception_handling(self):
        """Test handling of general exceptions during parsing."""
        # Given
        input_md = "Some content"

        # When
        with patch("jet.code.markdown_utils.read_md_content", return_value=input_md):
            with patch("jet.code.markdown_utils.MarkdownParser.parse", side_effect=Exception("Unexpected error")):
                # Then
                with pytest.raises(Exception, match="Unexpected error"):
                    parse_markdown("dummy_input", merge_paragraphs=True)

    def test_multi_level_headers(self):
        """Test parsing multiple headers with varying levels."""
        # Given
        input_md = "# Main Header\n## Sub Header\n### Sub Sub Header"
        expected = [
            {
                "type": "header",
                "content": "# Main Header",
                "level": 1,
                "meta": {},
                "line": 1
            },
            {
                "type": "header",
                "content": "## Sub Header",
                "level": 2,
                "meta": {},
                "line": 2
            },
            {
                "type": "header",
                "content": "### Sub Sub Header",
                "level": 3,
                "meta": {},
                "line": 3
            }
        ]

        # When
        with patch("jet.code.markdown_utils.read_md_content", return_value=input_md):
            with patch("jet.code.markdown_utils.MarkdownParser.parse", return_value=[
                {"type": "header", "content": "Main Header",
                    "level": 1, "meta": {}, "line": 1},
                {"type": "header", "content": "Sub Header",
                    "level": 2, "meta": {}, "line": 2},
                {"type": "header", "content": "Sub Sub Header",
                    "level": 3, "meta": {}, "line": 3}
            ]):
                result = parse_markdown("dummy_input", merge_paragraphs=True)

        # Then
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_complex_structure_sequence(self):
        """Test parsing a complex sequence: paragraphs -> non-paragraphs -> paragraph -> non-paragraph -> ordered list."""
        # Given
        input_md = """Para 1.\nPara 2.\n# Header\n```code\nprint('test')\n```\n- Item 1\n| Head | Col |\n|----|----|\n| R1C1 | R1C2 |\nPara 3\nPara 4\n> Blockquote\n1. Ordered item 1"""
        expected = [
            {
                "type": "paragraph",
                "content": "Para 1.\nPara 2.",
                "level": None,
                "meta": {},
                "line": 1
            },
            {
                "type": "header",
                "content": "# Header",
                "level": 1,
                "meta": {},
                "line": 3
            },
            {
                "type": "code",
                "content": "print('test')",
                "level": None,
                "meta": {"language": None, "code_type": "indented"},
                "line": 4
            },
            {
                "type": "unordered_list",
                "content": "- Item 1",
                "level": None,
                "meta": {"items": [{"text": "Item 1", "task_item": False}]},
                "line": 7
            },
            {
                "type": "table",
                "content": "| Head | Col  |\n| ---- | ---- |\n| R1C1 | R1C2 |",
                "level": None,
                "meta": {
                    "header": ["Head", "Col"],
                    "rows": [["R1C1", "R1C2"]]
                },
                "line": 8
            },
            {
                "type": "paragraph",
                "content": "Para 3\nPara 4",
                "level": None,
                "meta": {},
                "line": 10
            },
            {
                "type": "blockquote",
                "content": "> Blockquote",
                "level": None,
                "meta": {},
                "line": 11
            },
            {
                "type": "ordered_list",
                "content": "1. Ordered item 1",
                "level": None,
                "meta": {"items": [{"text": "Ordered item 1", "task_item": False}]},
                "line": 12
            }
        ]

        # When
        with patch("jet.code.markdown_utils.read_md_content", return_value=input_md):
            with patch("jet.code.markdown_utils.MarkdownParser.parse", return_value=[
                {"type": "paragraph", "content": "Para 1.",
                    "level": None, "meta": {}, "line": 1},
                {"type": "paragraph", "content": "Para 2.",
                    "level": None, "meta": {}, "line": 2},
                {"type": "header", "content": "Header",
                    "level": 1, "meta": {}, "line": 3},
                {"type": "code", "content": "print('test')", "level": None, "meta": {
                    "language": None, "code_type": "indented"}, "line": 4},
                {"type": "unordered_list", "content": "- Item 1", "level": None,
                    "meta": {"items": [{"text": "Item 1", "task_item": False}]}, "line": 7},
                {"type": "table", "content": "| Head | Col |\n|----|----|\n| R1C1 | R1C2 |", "level": None,
                    "meta": {"header": ["Head", "Col"], "rows": [["R1C1", "R1C2"]]}, "line": 8},
                {"type": "paragraph", "content": "Para 3\nPara 4",
                    "level": None, "meta": {}, "line": 10},
                {"type": "blockquote", "content": "> Blockquote",
                    "level": None, "meta": {}, "line": 11},
                {"type": "ordered_list", "content": "1. Ordered item 1", "level": None,
                    "meta": {"items": [{"text": "Ordered item 1", "task_item": False}]}, "line": 12}
            ]):
                result = parse_markdown("dummy_input", merge_paragraphs=True)

        # Then
        assert result == expected, f"Expected {expected}, but got {result}"
