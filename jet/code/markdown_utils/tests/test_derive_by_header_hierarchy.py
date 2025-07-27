import pytest
from typing import List
from jet.code.markdown_utils._markdown_parser import derive_by_header_hierarchy, HeaderDoc


class TestDeriveByHeaderHierarchy:
    def test_single_header_with_content(self):
        # Given: Markdown content with a single header and paragraph
        markdown_content = "# Header 1\nThis is some content."
        expected = [
            {
                "doc_index": 0,
                "doc_id": str,
                "header": "# Header 1",
                "content": "This is some content.",
                "level": 1,
                "parent_header": None,
                "parent_level": None,
                "tokens": [
                    {"type": "header", "content": "# Header 1",
                        "level": 1, "meta": {}, "line": int},
                    {"type": "paragraph", "content": "This is some content.",
                        "level": None, "meta": {}, "line": int}
                ]
            }
        ]

        # When: Parsing the markdown content
        result = derive_by_header_hierarchy(markdown_content)

        # Then: Result matches expected structure
        assert len(result) == len(expected)
        for res, exp in zip(result, expected):
            assert res["doc_index"] == exp["doc_index"]
            assert isinstance(res["doc_id"], str) and res["doc_id"]
            assert res["header"] == exp["header"]
            assert res["content"] == exp["content"]
            assert res["level"] == exp["level"]
            assert res["parent_header"] == exp["parent_header"]
            assert res["parent_level"] == exp["parent_level"]
            assert len(res["tokens"]) == len(exp["tokens"])
            for res_token, exp_token in zip(res["tokens"], exp["tokens"]):
                assert res_token["type"] == exp_token["type"]
                assert res_token["content"] == exp_token["content"]
                assert res_token["level"] == exp_token["level"]
                assert res_token["meta"] == exp_token["meta"]
                assert isinstance(res_token["line"], int)

    def test_nested_headers(self):
        # Given: Markdown with nested headers
        markdown_content = "# Header 1\nContent 1\n## Header 2\nContent 2"
        expected = [
            {
                "doc_index": 0,
                "doc_id": str,
                "header": "# Header 1",
                "content": "Content 1",
                "level": 1,
                "parent_header": None,
                "parent_level": None,
                "tokens": [
                    {"type": "header", "content": "# Header 1",
                        "level": 1, "meta": {}, "line": int},
                    {"type": "paragraph", "content": "Content 1",
                        "level": None, "meta": {}, "line": int}
                ]
            },
            {
                "doc_index": 1,
                "doc_id": str,
                "header": "## Header 2",
                "content": "Content 2",
                "level": 2,
                "parent_header": "# Header 1",
                "parent_level": 1,
                "tokens": [
                    {"type": "header", "content": "## Header 2",
                        "level": 2, "meta": {}, "line": int},
                    {"type": "paragraph", "content": "Content 2",
                        "level": None, "meta": {}, "line": int}
                ]
            }
        ]

        # When: Parsing the markdown content
        result = derive_by_header_hierarchy(markdown_content)

        # Then: Result matches expected structure with correct hierarchy
        assert len(result) == len(expected)
        for res, exp in zip(result, expected):
            assert res["doc_index"] == exp["doc_index"]
            assert isinstance(res["doc_id"], str) and res["doc_id"]
            assert res["header"] == exp["header"]
            assert res["content"] == exp["content"]
            assert res["level"] == exp["level"]
            assert res["parent_header"] == exp["parent_header"]
            assert res["parent_level"] == exp["parent_level"]
            assert len(res["tokens"]) == len(exp["tokens"])
            for res_token, exp_token in zip(res["tokens"], exp["tokens"]):
                assert res_token["type"] == exp_token["type"]
                assert res_token["content"] == exp_token["content"]
                assert res_token["level"] == exp_token["level"]
                assert res_token["meta"] == exp_token["meta"]
                assert isinstance(res_token["line"], int)

    def test_nested_headers_no_root_content(self):
        # Given: Markdown with nested headers
        markdown_content = "# Header 1\n## Header 2\nContent 2"
        expected = [
            {
                "doc_index": 0,
                "doc_id": str,
                "header": "## Header 2",
                "content": "Content 2",
                "level": 2,
                "parent_header": "# Header 1",
                "parent_level": 1,
                "tokens": [
                    {"type": "header", "content": "## Header 2",
                        "level": 2, "meta": {}, "line": int},
                    {"type": "paragraph", "content": "Content 2",
                        "level": None, "meta": {}, "line": int}
                ]
            }
        ]

        # When: Parsing the markdown content
        result = derive_by_header_hierarchy(markdown_content)

        # Then: Result matches expected structure with correct hierarchy
        assert len(result) == len(expected)
        for res, exp in zip(result, expected):
            assert res["doc_index"] == exp["doc_index"]
            assert isinstance(res["doc_id"], str) and res["doc_id"]
            assert res["header"] == exp["header"]
            assert res["content"] == exp["content"]
            assert res["level"] == exp["level"]
            assert res["parent_header"] == exp["parent_header"]
            assert res["parent_level"] == exp["parent_level"]
            assert len(res["tokens"]) == len(exp["tokens"])
            for res_token, exp_token in zip(res["tokens"], exp["tokens"]):
                assert res_token["type"] == exp_token["type"]
                assert res_token["content"] == exp_token["content"]
                assert res_token["level"] == exp_token["level"]
                assert res_token["meta"] == exp_token["meta"]
                assert isinstance(res_token["line"], int)

    def test_header_with_list(self):
        # Given: Markdown content with a header and unordered list
        markdown_content = "# Header 1\n- Item 1\n- Item 2"
        expected = [
            {
                "doc_index": 0,
                "doc_id": str,
                "header": "# Header 1",
                "content": "- Item 1\n- Item 2",
                "level": 1,
                "parent_header": None,
                "parent_level": None,
                "tokens": [
                    {"type": "header", "content": "# Header 1",
                        "level": 1, "meta": {}, "line": int},
                    {
                        "type": "unordered_list",
                        "content": "- Item 1\n- Item 2",
                        "level": None,
                        "meta": {"items": [{"text": "Item 1", "task_item": False}, {"text": "Item 2", "task_item": False}]},
                        "line": int
                    }
                ]
            }
        ]

        # When: Parsing the markdown content
        result = derive_by_header_hierarchy(markdown_content)

        # Then: Result matches expected structure with list content
        assert len(result) == len(expected)
        for res, exp in zip(result, expected):
            assert res["doc_index"] == exp["doc_index"]
            assert isinstance(res["doc_id"], str) and res["doc_id"]
            assert res["header"] == exp["header"]
            assert res["content"] == exp["content"]
            assert res["level"] == exp["level"]
            assert res["parent_header"] == exp["parent_header"]
            assert res["parent_level"] == exp["parent_level"]
            assert len(res["tokens"]) == len(exp["tokens"])
            for res_token, exp_token in zip(res["tokens"], exp["tokens"]):
                assert res_token["type"] == exp_token["type"]
                assert res_token["content"] == exp_token["content"]
                assert res_token["level"] == exp_token["level"]
                assert res_token["meta"] == exp_token["meta"]
                assert isinstance(res_token["line"], int)

    def test_empty_content(self):
        # Given: Empty markdown content
        markdown_content = ""
        expected: List[HeaderDoc] = []

        # When: Parsing the markdown content
        result = derive_by_header_hierarchy(markdown_content)

        # Then: Result is an empty list
        assert result == expected

    def test_no_headers(self):
        # Given: Markdown content with no headers
        markdown_content = "Just some content\nAnother line"
        expected = [
            {
                "doc_index": 0,
                "doc_id": str,
                "header": "",
                "content": "Just some content\nAnother line",
                "level": 0,
                "parent_header": None,
                "parent_level": None,
                "tokens": [
                    {"type": "paragraph", "content": "Just some content",
                        "level": None, "meta": {}, "line": int},
                    {"type": "paragraph", "content": "Another line",
                        "level": None, "meta": {}, "line": int}
                ]
            }
        ]

        # When: Parsing the markdown content
        result = derive_by_header_hierarchy(markdown_content)

        # Then: Result matches expected structure with default section
        assert len(result) == len(expected)
        for res, exp in zip(result, expected):
            assert res["doc_index"] == exp["doc_index"]
            assert isinstance(res["doc_id"], str) and res["doc_id"]
            assert res["header"] == exp["header"]
            assert res["content"] == exp["content"]
            assert res["level"] == exp["level"]
            assert res["parent_header"] == exp["parent_header"]
            assert res["parent_level"] == exp["parent_level"]
            assert len(res["tokens"]) == len(exp["tokens"])
            for res_token, exp_token in zip(res["tokens"], exp["tokens"]):
                assert res_token["type"] == exp_token["type"]
                assert res_token["content"] == exp_token["content"]
                assert res_token["level"] == exp_token["level"]
                assert res_token["meta"] == exp_token["meta"]
                assert isinstance(res_token["line"], int)
