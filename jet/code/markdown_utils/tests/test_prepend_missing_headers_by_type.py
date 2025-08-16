import pytest
from typing import List
from jet.code.markdown_types.markdown_parsed_types import MarkdownToken
from jet.code.markdown_utils._markdown_parser import prepend_missing_headers_by_type


class TestPrependMissingHeadersByType:
    def test_inserts_subheader_for_multiple_types(self):
        # Given: A list of markdown tokens with multiple types including a header, list, code, paragraphs, and blockquote
        tokens: List[MarkdownToken] = [
            {
                "content": "Header 1",
                "level": 2,
                "line": 1,
                "type": "header",
                "meta": {}
            },
            {
                "content": "This is a paragraph after the header.",
                "line": 2,
                "type": "paragraph",
                "level": None,
                "meta": {}
            },
            {
                "line": 3,
                "meta": {
                    "items": [
                        {
                            "text": "List item 1",
                            "task_item": False
                        }
                    ]
                },
                "type": "unordered_list",
                "content": "",
                "level": None
            },
            {
                "content": "def greet(name: str) -> str:\n return f\"Hello, {name}!\"",
                "line": 4,
                "meta": {
                    "language": "python"
                },
                "type": "code",
                "level": None
            },
            {
                "content": "Welcome to our project! This is an `introduction` to our work, featuring a [website](https://project.com).",
                "line": 5,
                "type": "paragraph",
                "level": None,
                "meta": {}
            },
            {
                "content": "![Project Logo](https://project.com/logo.png)",
                "line": 6,
                "type": "paragraph",
                "level": None,
                "meta": {}
            },
            {
                "content": "Note: Always check the [docs](https://docs.project.com) for updates.",
                "line": 7,
                "type": "blockquote",
                "level": None,
                "meta": {}
            },
        ]

        # When: We process the tokens to prepend missing headers
        result = prepend_missing_headers_by_type(tokens)

        # Then: The result should include new headers for non-paragraph, non-header tokens with correct line numbers
        expected = [
            {
                "content": "Header 1",
                "level": 2,
                "line": 1,
                "type": "header",
                "meta": {}
            },
            {
                "content": "This is a paragraph after the header.",
                "line": 2,
                "type": "paragraph",
                "level": None,
                "meta": {}
            },
            {
                "line": 3,
                "meta": {
                    "items": [
                        {
                            "text": "List item 1",
                            "task_item": False
                        }
                    ]
                },
                "type": "unordered_list",
                "content": "",
                "level": None
            },
            {
                "content": "Header 1",
                "line": 4,
                "type": "header",
                "level": 2,
                "meta": {}
            },
            {
                "content": "def greet(name: str) -> str:\n return f\"Hello, {name}!\"",
                "line": 5,
                "meta": {
                    "language": "python"
                },
                "type": "code",
                "level": None
            },
            {
                "content": "Header 1",
                "line": 6,
                "type": "header",
                "level": 2,
                "meta": {}
            },
            {
                "content": "Welcome to our project! This is an `introduction` to our work, featuring a [website](https://project.com).",
                "line": 7,
                "type": "paragraph",
                "level": None,
                "meta": {}
            },
            {
                "content": "![Project Logo](https://project.com/logo.png)",
                "line": 8,
                "type": "paragraph",
                "level": None,
                "meta": {}
            },
            {
                "content": "Note: Always check the [docs](https://docs.project.com) for updates.",
                "line": 9,
                "type": "blockquote",
                "level": None,
                "meta": {}
            }
        ]
        assert result == expected

    def test_replace_header_with_type(self):
        tokens: List[MarkdownToken] = [
            {
                "content": "Personal Information",
                "level": 2,
                "line": 1,
                "type": "header",
                "meta": {}
            },
            {
                "content": "Contact Details",
                "line": 2,
                "type": "header",
                "level": 3,
                "meta": {}
            },
            {
                "line": 3,
                "meta": {
                    "items": [
                        {
                            "text": "Full Name: Jethro Reuel A. Estrada",
                            "task_item": False
                        }
                    ]
                },
                "type": "unordered_list",
                "content": "",
                "level": None
            },
            {
                "content": "Personal Details",
                "line": 4,
                "type": "header",
                "level": 3,
                "meta": {}
            },
            {
                "line": 5,
                "meta": {},
                "type": "blockquote",
                "content": "Sample blockquote",
                "level": None
            },
            {
                "content": "Last paragraph",
                "line": 6,
                "type": "paragraph",
                "level": None,
                "meta": {}
            },
        ]

        result = prepend_missing_headers_by_type(tokens)

        expected = [
            {
                "type": "header",
                "content": "Personal Information",
                "level": 2,
                "meta": {},
                "line": 1
            },
            {
                "content": "Contact Details",
                "line": 2,
                "type": "header",
                "level": 3,
                "meta": {}
            },
            {
                "type": "unordered_list",
                "content": "",
                "level": None,
                "meta": {
                    "items": [
                        {
                            "text": "Full Name: Jethro Reuel A. Estrada",
                            "task_item": False
                        }
                    ]
                },
                "line": 3
            },
            {
                "content": "Personal Details",
                "line": 4,
                "type": "header",
                "level": 3,
                "meta": {}
            },
            {
                "line": 5,
                "meta": {},
                "type": "blockquote",
                "content": "Sample blockquote",
                "level": None
            },
            {
                "type": "paragraph",
                "content": "Last paragraph",
                "level": None,
                "meta": {},
                "line": 6
            },
        ]

        assert result == expected
