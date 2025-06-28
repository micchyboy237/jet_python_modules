from typing import List, Dict, Any
import pytest

# Assuming MarkdownToken is a TypedDict
from typing import TypedDict

from jet.code.markdown_types import MarkdownToken
from jet.code.markdown_utils import remove_list_table_placeholders


class TestRemoveListTablePlaceholders:
    def test_remove_placeholder_header_and_following_non_headers(self):
        # Given: A list of tokens with a placeholder header and following non-header tokens
        given_tokens: List[MarkdownToken] = [
            {"type": "header", "content": "Main Header",
                "level": 1, "meta": {}, "line": 1},
            {"type": "paragraph", "content": "Some text",
                "level": None, "meta": {}, "line": 2},
            {"type": "header", "content": "# placeholder",
                "level": 2, "meta": {}, "line": 3},
            {"type": "paragraph", "content": "Placeholder text",
                "level": None, "meta": {}, "line": 4},
            {"type": "unordered_list", "content": "- Item", "level": None,
                "meta": {"items": [{"text": "Item"}]}, "line": 5},
            {"type": "header", "content": "Next Section",
                "level": 2, "meta": {}, "line": 6},
            {"type": "paragraph", "content": "More text",
                "level": None, "meta": {}, "line": 7}
        ]
        expected: List[MarkdownToken] = [
            {"type": "header", "content": "Main Header",
                "level": 1, "meta": {}, "line": 1},
            {"type": "paragraph", "content": "Some text",
                "level": None, "meta": {}, "line": 2},
            {"type": "header", "content": "Next Section",
                "level": 2, "meta": {}, "line": 6},
            {"type": "paragraph", "content": "More text",
                "level": None, "meta": {}, "line": 7}
        ]

        # When: The function is called to remove placeholder headers and their following non-header tokens
        result = remove_list_table_placeholders(given_tokens)

        # Then: The result should match the expected tokens, excluding the placeholder header and its non-header tokens
        assert result == expected

    def test_no_placeholder_header(self):
        # Given: A list of tokens without any placeholder headers
        given_tokens: List[MarkdownToken] = [
            {"type": "header", "content": "Main Header",
                "level": 1, "meta": {}, "line": 1},
            {"type": "paragraph", "content": "Some text",
                "level": None, "meta": {}, "line": 2},
            {"type": "header", "content": "Next Section",
                "level": 2, "meta": {}, "line": 3}
        ]
        expected: List[MarkdownToken] = given_tokens  # No changes expected

        # When: The function is called
        result = remove_list_table_placeholders(given_tokens)

        # Then: The result should match the input tokens exactly
        assert result == expected

    def test_multiple_placeholder_headers(self):
        # Given: A list of tokens with multiple placeholder headers
        given_tokens: List[MarkdownToken] = [
            {"type": "header", "content": "Main Header",
                "level": 1, "meta": {}, "line": 1},
            {"type": "header", "content": "# placeholder",
                "level": 2, "meta": {}, "line": 2},
            {"type": "paragraph", "content": "Text to remove",
                "level": None, "meta": {}, "line": 3},
            {"type": "header", "content": "# placeholder",
                "level": 2, "meta": {}, "line": 4},
            {"type": "unordered_list", "content": "- Item", "level": None,
                "meta": {"items": [{"text": "Item"}]}, "line": 5},
            {"type": "header", "content": "Final Section",
                "level": 2, "meta": {}, "line": 6}
        ]
        expected: List[MarkdownToken] = [
            {"type": "header", "content": "Main Header",
                "level": 1, "meta": {}, "line": 1},
            {"type": "header", "content": "Final Section",
                "level": 2, "meta": {}, "line": 6}
        ]

        # When: The function is called
        result = remove_list_table_placeholders(given_tokens)

        # Then: The result should exclude all placeholder headers and their following non-header tokens
        assert result == expected

    def test_placeholder_header_at_end(self):
        # Given: A list of tokens with a placeholder header at the end
        given_tokens: List[MarkdownToken] = [
            {"type": "header", "content": "Main Header",
                "level": 1, "meta": {}, "line": 1},
            {"type": "paragraph", "content": "Some text",
                "level": None, "meta": {}, "line": 2},
            {"type": "header", "content": "# placeholder",
                "level": 2, "meta": {}, "line": 3},
            {"type": "paragraph", "content": "Text to remove",
                "level": None, "meta": {}, "line": 4}
        ]
        expected: List[MarkdownToken] = [
            {"type": "header", "content": "Main Header",
                "level": 1, "meta": {}, "line": 1},
            {"type": "paragraph", "content": "Some text",
                "level": None, "meta": {}, "line": 2}
        ]

        # When: The function is called
        result = remove_list_table_placeholders(given_tokens)

        # Then: The result should exclude the placeholder header and its following non-header tokens
        assert result == expected
