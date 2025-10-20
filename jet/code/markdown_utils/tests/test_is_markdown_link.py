# jet_python_modules/jet/code/markdown_utils/tests/test_is_markdown_link.py
import pytest
from jet.code.markdown_utils._preprocessors import is_markdown_link


class TestIsMarkdownLink:
    """Test suite for the is_markdown_link function."""

    @pytest.mark.parametrize(
        "input_text, expected",
        [
            # Single text links
            ("[Google](https://www.google.com)", True),
            ("[Link](http://example.com)", True),
            ("[](/path)", True),
            # Single image links
            ("![alt](https://example.com/image.jpg)", True),
            ("![](/image.png)", True),
            # Plain URLs (treated as links)
            ("https://www.google.com", True),
            ("http://example.com/path", True),
            # Empty string
            ("", True),
            # Whitespace only
            ("   ", True),
        ],
    )
    def test_single_link_variations(self, input_text: str, expected: bool) -> None:
        """
        Given: Various single markdown links, image links, or plain URLs
        When: is_markdown_link is called
        Then: Returns True
        """
        result = is_markdown_link(input_text)
        assert result == expected

    @pytest.mark.parametrize(
        "input_text, expected",
        [
            # Text with links
            ("Visit [Google](https://www.google.com) now", False),
            ("Check [site](https://site.com)", False),
            ("Text before ![img](image.png)", False),
            # Multiple links
            ("[link1](a.com) [link2](b.com)", False),
            ("![img1](a.jpg) ![img2](b.jpg)", False),
            # Mixed content
            ("[link](url) and plain text", False),
            # Plain text only
            ("Just plain text here", False),
            # Multiline with content
            ("""Line 1
Line 2 with [link](url)""", False),
        ],
    )
    def test_content_with_links(self, input_text: str, expected: bool) -> None:
        """
        Given: Text containing links mixed with other content
        When: is_markdown_link is called
        Then: Returns False
        """
        result = is_markdown_link(input_text)
        assert result == expected

    @pytest.mark.parametrize(
        "input_text, expected",
        [
            # Nested brackets in links
            ("[nested [brackets]](https://example.com)", True),
            ("![alt [with brackets]](image.jpg)", True),
            # Links with spaces
            (" [Google](https://google.com) ", True),
            ("![ alt ]( image.jpg )", True),
            # Edge cases
            ("[ ] (url)", True),
            ("! [ ] (image.jpg)", True),
        ],
    )
    def test_edge_case_links(self, input_text: str, expected: bool) -> None:
        """
        Given: Edge case markdown links with nested brackets or extra whitespace
        When: is_markdown_link is called
        Then: Returns True for valid single links
        """
        result = is_markdown_link(input_text)
        assert result == expected

    def test_multiline_single_link(self) -> None:
        """
        Given: Multiline text that is just a single link
        When: is_markdown_link is called
        Then: Returns True
        """
        input_text = """[Google](https://www.google.com)"""
        expected = True
        result = is_markdown_link(input_text)
        assert result == expected

    def test_multiline_with_only_whitespace(self) -> None:
        """
        Given: Multiline text with only whitespace
        When: is_markdown_link is called
        Then: Returns True
        """
        input_text = """   
        
        """
        expected = True
        result = is_markdown_link(input_text)
        assert result == expected