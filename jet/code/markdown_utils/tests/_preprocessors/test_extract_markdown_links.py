from typing import List, Optional, TypedDict
import pytest

from jet.code.markdown_utils._preprocessors import extract_markdown_links, MDHeaderLink


@pytest.fixture
def test_base_url():
    return "https://example.com"


class TestExtractMarkdownLinks:
    def test_extract_simple_markdown_link(self):
        text = "This is a [link](https://example.com/page) in text"
        expected_links = [{
            "text": "link",
            "url": "https://example.com/page",
            "start_idx": 10,
            "end_idx": 42,
            "line": "This is a [link](https://example.com/page) in text",
            "line_idx": 0,
            "is_heading": False
        }]
        expected_output = "This is a link in text"
        result_links, result_output = extract_markdown_links(
            text, ignore_links=True)
        assert result_links == expected_links
        assert result_output == expected_output

    def test_extract_nested_image_link(self):
        text = "See this [![alt](image.jpg)](https://example.com)"
        expected_links = [{
            "text": "alt",
            "url": "https://example.com",
            "start_idx": 9,
            "end_idx": 49,
            "line": "See this [![alt](image.jpg)](https://example.com)",
            "line_idx": 0,
            "is_heading": False
        }]
        expected_output = "See this alt"
        result_links, result_output = extract_markdown_links(
            text, ignore_links=True)
        assert result_links == expected_links
        assert result_output == expected_output

    def test_extract_plain_url(self):
        text = "Visit https://example.com for more info"
        expected_links = [{
            "text": "",
            "url": "https://example.com",
            "start_idx": 6,
            "end_idx": 25,
            "line": "Visit https://example.com for more info",
            "line_idx": 0,
            "is_heading": False
        }]
        expected_output = "Visit  for more info"
        result_links, result_output = extract_markdown_links(
            text, ignore_links=True)
        assert result_links == expected_links
        assert result_output == expected_output

    def test_preserve_links_when_ignore_links_false(self):
        text = "This is a [link](https://example.com)"
        expected_links = [{
            "text": "link",
            "url": "https://example.com",
            "start_idx": 10,
            "end_idx": 37,
            "line": "This is a [link](https://example.com)",
            "line_idx": 0,
            "is_heading": False
        }]
        expected_output = "This is a [link](https://example.com)"
        result_links, result_output = extract_markdown_links(
            text, ignore_links=False)
        assert result_links == expected_links
        assert result_output == expected_output

    def test_resolve_relative_url_with_base_url(self, test_base_url):
        text = "Check [page](/about) for details"
        expected_links = [{
            "text": "page",
            "url": "https://example.com/about",
            "start_idx": 6,
            "end_idx": 20,
            "line": "Check [page](/about) for details",
            "line_idx": 0,
            "is_heading": False
        }]
        expected_output = "Check page for details"
        result_links, result_output = extract_markdown_links(
            text, base_url=test_base_url, ignore_links=True)
        assert result_links == expected_links
        assert result_output == expected_output

    def test_heading_detection(self):
        text = "# [Heading Link](https://example.com)"
        expected_links = [{
            "text": "Heading Link",
            "url": "https://example.com",
            "start_idx": 2,
            "end_idx": 37,
            "line": "# [Heading Link](https://example.com)",
            "line_idx": 0,
            "is_heading": True
        }]
        expected_output = "# Heading Link"
        result_links, result_output = extract_markdown_links(
            text, ignore_links=True)
        assert result_links == expected_links
        assert result_output == expected_output
