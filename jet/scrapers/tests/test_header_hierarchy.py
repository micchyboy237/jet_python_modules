from typing import List
import pytest
from jet.scrapers.header_hierarchy import extract_header_hierarchy, HeaderDoc
from jet.scrapers.utils import BaseNode


class TestExtractHeaderHierarchy:
    def test_single_header_with_content(self):
        # Given: HTML with a single header and content
        html = """
        <h1>Main Header</h1>
        <p>This is some content.</p>
        """
        # When: Extracting header hierarchy
        result: List[HeaderDoc] = extract_header_hierarchy(html)
        # Then: Expect one section with header and content HTML combined
        expected: List[HeaderDoc] = [{
            "id": result[0]["id"],
            "doc_index": 0,
            "header": "Main Header",
            "content": "This is some content.",
            "level": 1,
            "depth": 1,
            "parent_headers": [],
            "parent_header": None,
            "parent_level": None,
            "html": "<h1>Main Header</h1>\n<p>This is some content.</p>",
            "tag": "h1"
        }]
        assert len(result) == 1
        assert result[0]["header"] == expected[0]["header"]
        assert result[0]["content"] == expected[0]["content"]
        assert result[0]["level"] == expected[0]["level"]
        assert result[0]["depth"] == expected[0]["depth"]
        assert result[0]["parent_headers"] == expected[0]["parent_headers"]
        assert result[0]["parent_header"] == expected[0]["parent_header"]
        assert result[0]["parent_level"] == expected[0]["parent_level"]
        assert result[0]["html"] == expected[0]["html"]
        assert result[0]["tag"] == expected[0]["tag"]

    def test_nested_headers(self):
        # Given: HTML with nested headers and content
        html = """
        <h1>Main Header</h1>
        <p>Main content.</p>
        <h2 class="sub">Sub Header</h2>
        <p>Sub content.</p>
        """
        # When: Extracting header hierarchy
        result: List[HeaderDoc] = extract_header_hierarchy(html)
        # Then: Expect two sections with header and content HTML combined
        expected: List[HeaderDoc] = [
            {
                "id": result[0]["id"],
                "doc_index": 0,
                "header": "Main Header",
                "content": "Main content.",
                "level": 1,
                "depth": 1,
                "parent_headers": [],
                "parent_header": None,
                "parent_level": None,
                "html": "<h1>Main Header</h1>\n<p>Main content.</p>",
                "tag": "h1"
            },
            {
                "id": result[1]["id"],
                "doc_index": 1,
                "header": "Sub Header",
                "content": "Sub content.",
                "level": 2,
                "depth": 1,
                "parent_headers": ["Main Header"],
                "parent_header": "Main Header",
                "parent_level": 1,
                "html": '<h2 class="sub">Sub Header</h2>\n<p>Sub content.</p>',
                "tag": "h2"
            }
        ]
        assert len(result) == 2
        for i in range(2):
            assert result[i]["header"] == expected[i]["header"]
            assert result[i]["content"] == expected[i]["content"]
            assert result[i]["level"] == expected[i]["level"]
            assert result[i]["depth"] == expected[i]["depth"]
            assert result[i]["parent_headers"] == expected[i]["parent_headers"]
            assert result[i]["parent_header"] == expected[i]["parent_header"]
            assert result[i]["parent_level"] == expected[i]["parent_level"]
            assert result[i]["html"] == expected[i]["html"]
            assert result[i]["tag"] == expected[i]["tag"]

    def test_content_before_header(self):
        # Given: HTML with content before a header
        html = """
        <p>Intro content.</p>
        <h1>Main Header</h1>
        <p>Main content.</p>
        """
        # When: Extracting header hierarchy
        result: List[HeaderDoc] = extract_header_hierarchy(html)
        # Then: Expect one section for the header with content, ignoring content before header
        expected: List[HeaderDoc] = [
            {
                "id": result[0]["id"],
                "doc_index": 0,
                "header": "Main Header",
                "content": "Main content.",
                "level": 1,
                "depth": 1,
                "parent_headers": [],
                "parent_header": None,
                "parent_level": None,
                "html": "<h1>Main Header</h1>\n<p>Main content.</p>",
                "tag": "h1"
            }
        ]
        assert len(result) == 1
        assert result[0]["header"] == expected[0]["header"]
        assert result[0]["content"] == expected[0]["content"]
        assert result[0]["level"] == expected[0]["level"]
        assert result[0]["depth"] == expected[0]["depth"]
        assert result[0]["parent_headers"] == expected[0]["parent_headers"]
        assert result[0]["parent_header"] == expected[0]["parent_header"]
        assert result[0]["parent_level"] == expected[0]["parent_level"]
        assert result[0]["html"] == expected[0]["html"]
        assert result[0]["tag"] == expected[0]["tag"]

    def test_excluded_elements(self):
        # Given: HTML with excluded elements like script and footer
        html = """
        <h1>Main Header</h1>
        <p>Main content.</p>
        <script>alert('test');</script>
        <footer>Footer content</footer>
        """
        # When: Extracting header hierarchy
        result: List[HeaderDoc] = extract_header_hierarchy(html)
        # Then: Expect one section with header and content HTML, excluding script and footer
        expected: List[HeaderDoc] = [{
            "id": result[0]["id"],
            "doc_index": 0,
            "header": "Main Header",
            "content": "Main content.",
            "level": 1,
            "depth": 1,
            "parent_headers": [],
            "parent_header": None,
            "parent_level": None,
            "html": "<h1>Main Header</h1>\n<p>Main content.</p>",
            "tag": "h1"
        }]
        assert len(result) == 1
        assert result[0]["header"] == expected[0]["header"]
        assert result[0]["content"] == expected[0]["content"]
        assert result[0]["level"] == expected[0]["level"]
        assert result[0]["depth"] == expected[0]["depth"]
        assert result[0]["parent_headers"] == expected[0]["parent_headers"]
        assert result[0]["parent_header"] == expected[0]["parent_header"]
        assert result[0]["parent_level"] == expected[0]["parent_level"]
        assert result[0]["html"] == expected[0]["html"]
        assert result[0]["tag"] == expected[0]["tag"]
        assert "alert('test')" not in result[0]["content"]
        assert "Footer content" not in result[0]["content"]

    def test_nested_elements(self):
        # Given: HTML with a header inside a nested element
        html = """
        <h1>Main Header</h1>
        <div>
            <p>Nested content.</p>
            <h2>Nested Sub Header</h2>
            <p>Sub content.</p>
        </div>
        """
        # When: Extracting header hierarchy
        result: List[HeaderDoc] = extract_header_hierarchy(html)
        # Then: Expect two sections, one for the main header with nested content, and one for the nested header
        expected: List[HeaderDoc] = [
            {
                "id": result[0]["id"],
                "doc_index": 0,
                "header": "Main Header",
                "content": "Nested content.",
                "level": 1,
                "depth": 1,
                "parent_headers": [],
                "parent_header": None,
                "parent_level": None,
                "html": "<h1>Main Header</h1>\n<p>Nested content.</p>",
                "tag": "h1"
            },
            {
                "id": result[1]["id"],
                "doc_index": 1,
                "header": "Nested Sub Header",
                "content": "Sub content.",
                "level": 2,
                "depth": 2,
                "parent_headers": ["Main Header"],
                "parent_header": "Main Header",
                "parent_level": 1,
                "html": "<h2>Nested Sub Header</h2>\n<p>Sub content.</p>",
                "tag": "h2"
            }
        ]
        assert len(result) == 2
        for i in range(2):
            assert result[i]["header"] == expected[i]["header"]
            assert result[i]["content"] == expected[i]["content"]
            assert result[i]["level"] == expected[i]["level"]
            assert result[i]["depth"] == expected[i]["depth"]
            assert result[i]["parent_headers"] == expected[i]["parent_headers"]
            assert result[i]["parent_header"] == expected[i]["parent_header"]
            assert result[i]["parent_level"] == expected[i]["parent_level"]
            assert result[i]["html"] == expected[i]["html"]
            assert result[i]["tag"] == expected[i]["tag"]
