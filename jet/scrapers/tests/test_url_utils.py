import pytest
import re
from typing import Tuple, List
from jet.code.splitter_markdown_utils import extract_markdown_links
from jet.scrapers.utils import protect_links, restore_links


class TestProtectLinks:
    def test_basic_markdown_link(self):
        # Given: Text with a single markdown link
        input_text = "Content with [link](http://example.com)"
        expected_text = re.sub(
            r'\[link\]\(http://example\.com\)', '__LINK_0_[0-9a-f]{8}__', input_text)
        expected_links = ["[link](http://example.com)"]

        # When: Protect links is called
        result_text, result_links = protect_links(input_text)

        # Then: The text is replaced with a placeholder, and the link is captured
        assert re.match(expected_text, result_text)
        assert result_links == expected_links

    def test_markdown_link_with_caption(self):
        # Given: Text with a markdown link and caption
        input_text = 'Content with [link](http://example.com "Caption")'
        expected_text = re.sub(
            r'\[link\]\(http://example\.com "Caption"\)', '__LINK_0_[0-9a-f]{8}__', input_text)
        expected_links = ['[link](http://example.com "Caption")']

        # When: Protect links is called
        result_text, result_links = protect_links(input_text)

        # Then: The text is replaced with a placeholder, and the link with caption is captured
        assert re.match(expected_text, result_text)
        assert result_links == expected_links

    def test_plain_url(self):
        # Given: Text with a plain URL
        input_text = "Visit http://example.com/page?q=1#fragment"
        expected_text = re.sub(
            r'http://example\.com/page\?q=1#fragment', '__LINK_0_[0-9a-f]{8}__', input_text)
        expected_links = ["http://example.com/page?q=1#fragment"]

        # When: Protect links is called
        result_text, result_links = protect_links(input_text)

        # Then: The URL is replaced with a placeholder, and the URL is captured
        assert re.match(expected_text, result_text)
        assert result_links == expected_links

    def test_multiple_links(self):
        # Given: Text with multiple markdown and plain URLs
        input_text = (
            "See [link1](http://example.com) and http://plain.com "
            "and [link2](http://test.com \"Test\")"
        )
        expected_text = re.sub(
            r'\[link1\]\(http://example\.com\)', '__LINK_0_[0-9a-f]{8}__', input_text
        )
        expected_text = re.sub(
            r'http://plain\.com', '__LINK_1_[0-9a-f]{8}__', expected_text
        )
        expected_text = re.sub(
            r'\[link2\]\(http://test\.com "Test"\)', '__LINK_2_[0-9a-f]{8}__', expected_text
        )
        expected_links = [
            "[link1](http://example.com)",
            '[link2](http://test.com "Test")',
            "http://plain.com",
        ]

        # When: Protect links is called
        result_text, result_links = protect_links(input_text)

        # Then: All links are replaced with unique placeholders, and links are captured
        assert re.match(expected_text, result_text)
        assert result_links == expected_links

    def test_empty_input(self):
        # Given: Empty input text
        input_text = ""
        expected_text = ""
        expected_links: List[str] = []

        # When: Protect links is called
        result_text, result_links = protect_links(input_text)

        # Then: Empty text and empty link list are returned
        assert result_text == expected_text
        assert result_links == expected_links

    def test_no_links(self):
        # Given: Text without any links
        input_text = "Just plain text with no URLs."
        expected_text = input_text
        expected_links: List[str] = []

        # When: Protect links is called
        result_text, result_links = protect_links(input_text)

        # Then: Original text is unchanged, and no links are captured
        assert result_text == expected_text
        assert result_links == expected_links

    def test_malformed_markdown_link(self):
        # Given: Text with a malformed markdown link
        input_text = "Content with [link](http://example.com"
        expected_text = input_text  # Malformed link should be ignored
        expected_links: List[str] = []

        # When: Protect links is called
        result_text, result_links = protect_links(input_text)

        # Then: Text remains unchanged, and no links are captured
        assert result_text == expected_text
        assert result_links == expected_links

    def test_special_characters_in_url(self):
        # Given: Text with a URL containing special characters
        input_text = "Link: http://example.com/path?a=1&b=2#section"
        expected_text = re.sub(
            r'http://example\.com/path\?a=1&b=2#section', '__LINK_0_[0-9a-f]{8}__', input_text
        )
        expected_links = ["http://example.com/path?a=1&b=2#section"]

        # When: Protect links is called
        result_text, result_links = protect_links(input_text)

        # Then: URL is replaced with a placeholder, and the URL is captured
        assert re.match(expected_text, result_text)
        assert result_links == expected_links


class TestRestoreLinks:
    def test_restore_single_link(self):
        # Given: Protected text with a single placeholder and link
        protected_text = "Content with __LINK_0_12345678__"
        links = ["[link](http://example.com)"]
        expected_text = "Content with [link](http://example.com)"

        # When: Restore links is called
        result_text = restore_links(protected_text, links)

        # Then: The placeholder is replaced with the original link
        assert result_text == expected_text

    def test_restore_link_with_caption(self):
        # Given: Protected text with a placeholder for a link with caption
        protected_text = "Content with __LINK_0_12345678__"
        links = ['[link](http://example.com "Caption")']
        expected_text = 'Content with [link](http://example.com "Caption")'

        # When: Restore links is called
        result_text = restore_links(protected_text, links)

        # Then: The placeholder is replaced with the link and caption
        assert result_text == expected_text

    def test_restore_plain_url(self):
        # Given: Protected text with a placeholder for a plain URL
        protected_text = "Visit __LINK_0_12345678__"
        links = ["http://example.com/page?q=1#fragment"]
        expected_text = "Visit http://example.com/page?q=1#fragment"

        # When: Restore links is called
        result_text = restore_links(protected_text, links)

        # Then: The placeholder is replaced with the plain URL
        assert result_text == expected_text

    def test_restore_multiple_links(self):
        # Given: Protected text with multiple placeholders and links
        protected_text = (
            "See __LINK_0_12345678__ and __LINK_1_87654321__ "
            "and __LINK_2_abcdef12__"
        )
        links = [
            "[link1](http://example.com)",
            "http://plain.com",
            '[link2](http://test.com "Test")',
        ]
        expected_text = (
            "See [link1](http://example.com) and http://plain.com "
            'and [link2](http://test.com "Test")'
        )

        # When: Restore links is called
        result_text = restore_links(protected_text, links)

        # Then: All placeholders are replaced with their respective links
        assert result_text == expected_text

    def test_restore_no_placeholders(self):
        # Given: Text with no placeholders
        protected_text = "Just plain text with no links."
        links: List[str] = []
        expected_text = protected_text

        # When: Restore links is called
        result_text = restore_links(protected_text, links)

        # Then: Text remains unchanged
        assert result_text == expected_text

    def test_restore_empty_input(self):
        # Given: Empty protected text and empty links
        protected_text = ""
        links: List[str] = []
        expected_text = ""

        # When: Restore links is called
        result_text = restore_links(protected_text, links)

        # Then: Empty text is returned
        assert result_text == expected_text

    def test_restore_unmatched_placeholders(self):
        # Given: Protected text with a placeholder but no corresponding link
        protected_text = "Content with __LINK_0_12345678__"
        links: List[str] = []
        expected_text = protected_text

        # When: Restore links is called
        result_text = restore_links(protected_text, links)

        # Then: Text remains unchanged
        assert result_text == expected_text

    def test_restore_special_characters(self):
        # Given: Protected text with a placeholder for a URL with special characters
        protected_text = "Link: __LINK_0_12345678__"
        links = ["http://example.com/path?a=1&b=2#section"]
        expected_text = "Link: http://example.com/path?a=1&b=2#section"

        # When: Restore links is called
        result_text = restore_links(protected_text, links)

        # Then: The placeholder is replaced with the URL including special characters
        assert result_text == expected_text

    def test_restore_overlapping_placeholders(self):
        # Given: Protected text with potentially overlapping placeholders
        protected_text = "Links: __LINK_0_12345678____LINK_1_87654321__"
        links = ["[link1](http://example.com)", "[link2](http://test.com)"]
        expected_text = "Links: [link1](http://example.com)[link2](http://test.com)"

        # When: Restore links is called
        result_text = restore_links(protected_text, links)

        # Then: Placeholders are correctly replaced without overlap issues
        assert result_text == expected_text


class TestExtractMarkdownLinks:
    def test_ignore_links_true(self):
        # Given: Text with markdown and plain URLs
        input_text = "See [link1](http://example.com) and http://plain.com"
        expected_text = "See link1 and "
        expected_links: List[dict] = []

        # When: Extract links with ignore_links=True
        result_links, result_text = extract_markdown_links(
            input_text, ignore_links=True)

        # Then: Links are removed, markdown link replaced with label, plain URL removed
        assert result_text == expected_text
        assert result_links == expected_links

    def test_ignore_links_false(self):
        # Given: Text with markdown and plain URLs
        input_text = "See [link1](http://example.com) and http://plain.com"
        expected_text = "See [link1](http://example.com) and http://plain.com"
        expected_links = [
            {
                "text": "link1",
                "url": "http://example.com",
                "caption": None,
                "start_idx": 4,
                "end_idx": 27,
                "line": "[link1](http://example.com)",
                "line_idx": 0,
                "is_heading": True
            },
            {
                "text": "",
                "url": "http://plain.com",
                "caption": None,
                "start_idx": 31,
                "end_idx": 47,
                "line": "http://plain.com",
                "line_idx": 0,
                "is_heading": True
            }
        ]

        # When: Extract links with ignore_links=False
        result_links, result_text = extract_markdown_links(
            input_text, ignore_links=False)

        # Then: Links are extracted with cleaned URLs, text retains original links
        assert result_text == expected_text
        assert len(result_links) == len(expected_links)
        for result, expected in zip(result_links, expected_links):
            assert result["text"] == expected["text"]
            assert result["url"] == expected["url"]
            assert result["caption"] == expected["caption"]
            assert result["line_idx"] == expected["line_idx"]
            assert result["is_heading"] == expected["is_heading"]

    def test_ignore_links_true_with_caption(self):
        # Given: Text with a markdown link with caption
        input_text = 'Content with [link](http://example.com "Caption")'
        expected_text = "Content with link"
        expected_links: List[dict] = []

        # When: Extract links with ignore_links=True
        result_links, result_text = extract_markdown_links(
            input_text, ignore_links=True)

        # Then: Markdown link is replaced with label, no links returned
        assert result_text == expected_text
        assert result_links == expected_links

    def test_ignore_links_false_with_base_url(self):
        # Given: Text with a relative markdown link and base URL
        input_text = "See [link1](/path/page)"
        base_url = "http://example.com"
        expected_text = "See link1"
        expected_links = [
            {
                "text": "link1",
                "url": "http://example.com/path/page",
                "caption": None,
                "start_idx": 4,
                "end_idx": 22,
                "line": "[link1](http://example.com/path/page)",
                "line_idx": 0,
                "is_heading": True
            }
        ]

        # When: Extract links with ignore_links=False and base_url
        result_links, result_text = extract_markdown_links(
            input_text, base_url=base_url, ignore_links=False)

        # Then: Relative URL is resolved, link extracted, text replaced with label
        assert result_text == expected_text
        assert len(result_links) == len(expected_links)
        for result, expected in zip(result_links, expected_links):
            assert result["text"] == expected["text"]
            assert result["url"] == expected["url"]
            assert result["caption"] == expected["caption"]
            assert result["line_idx"] == expected["line_idx"]
            assert result["is_heading"] == expected["is_heading"]

    def test_empty_input(self):
        # Given: Empty input text
        input_text = ""
        expected_text = ""
        expected_links: List[dict] = []

        # When: Extract links with ignore_links=True
        result_links, result_text = extract_markdown_links(
            input_text, ignore_links=True)

        # Then: Empty text and links returned
        assert result_text == expected_text
        assert result_links == expected_links

    def test_no_links(self):
        # Given: Text with no links
        input_text = "Just plain text"
        expected_text = "Just plain text"
        expected_links: List[dict] = []

        # When: Extract links with ignore_links=True
        result_links, result_text = extract_markdown_links(
            input_text, ignore_links=True)

        # Then: Text unchanged, no links returned
        assert result_text == expected_text
        assert result_links == expected_links
