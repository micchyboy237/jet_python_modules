from jet.scrapers.utils import clean_newlines, clean_text, clean_spaces
from jet.code.splitter_markdown_utils import get_md_header_contents
import pytest
import unittest


class TestGetMdHeaderContents:
    @pytest.fixture
    def sample_md(self) -> str:
        return """
# Main Header
Some content under main header.

## Sub Header 1
Content for sub header 1.

### Sub Sub Header
Content for sub sub header.

## Sub Header 2
More content.
"""

    @pytest.fixture
    def sample_html(self) -> str:
        return """
<html>
<body>
<h1>Main Header</h1>
<p>Some content under main header.</p>
<h2>Sub Header 1</h2>
<p>Content for sub header 1.</p>
<h3>Sub Sub Header</h3>
<p>Content for sub sub header.</p>
<h2>Sub Header 2</h2>
<p>More content.</p>
</body>
</html>
"""

    def test_basic_markdown_headers(self, sample_md: str):
        headers = get_md_header_contents(sample_md)
        assert len(headers) == 4, "Should detect all headers"
        assert headers[0]["header"] == "# Main Header"
        assert headers[0]["header_level"] == 1
        assert headers[0]["parent_header"] is None
        assert headers[1]["header"] == "## Sub Header 1"
        assert headers[1]["header_level"] == 2
        assert headers[1]["parent_header"] == "# Main Header"
        assert headers[2]["header"] == "### Sub Sub Header"
        assert headers[2]["header_level"] == 3
        assert headers[2]["parent_header"] == "## Sub Header 1"
        assert headers[3]["header"] == "## Sub Header 2"
        assert headers[3]["header_level"] == 2
        assert headers[3]["parent_header"] == "# Main Header"

    def test_html_input(self, sample_html: str):
        headers = get_md_header_contents(sample_html, ignore_links=True)
        assert len(
            headers) >= 4, "Should convert HTML to markdown and detect headers"
        assert any(
            h["header"] == "# Main Header" for h in headers), "Should detect h1 as # Main Header"
        assert any(
            h["header"] == "## Sub Header 1" for h in headers), "Should detect h2 as ## Sub Header 1"

    def test_custom_headers_to_split_on(self, sample_md: str):
        headers_to_split_on = [("#", "h1"), ("##", "h2")]
        headers = get_md_header_contents(
            sample_md, headers_to_split_on=headers_to_split_on)
        assert len(headers) == 3, "Should only split on # and ## headers"
        assert all(h["header_level"] <=
                   2 for h in headers), "Should not include headers beyond level 2"

    def test_content_extraction(self, sample_md: str):
        headers = get_md_header_contents(sample_md)
        for header in headers:
            if header["header"] == "# Main Header":
                assert header["content"] == "Some content under main header.", "Content should match header section"
            elif header["header"] == "## Sub Header 1":
                assert header["content"] == "Content for sub header 1.", "Content should match sub header section"

    def test_empty_input(self):
        headers = get_md_header_contents("")
        assert headers == [], "Should return empty list for empty input"

    def test_invalid_header(self):
        md_text = """
Not a header
#Valid Header
Content
"""
        headers = get_md_header_contents(md_text)
        assert len(headers) == 1, "Should only process valid headers"
        assert headers[0]["header"] == "#Valid Header"

    def test_cleaning_functions(self, sample_md: str):
        headers = get_md_header_contents(sample_md)
        for header in headers:
            assert header["header"] == clean_spaces(
                header["header"]), "Header should be cleaned of extra spaces"
            assert header["content"] == clean_newlines(clean_text(
                header["content"]), max_newlines=1, strip_lines=True), "Content should be cleaned"

    def test_italic_headers(self):
        md_text = """
* # Italic Main Header
Content under italic header.

* ## Italic Sub Header
Sub content.
"""
        headers = get_md_header_contents(md_text)
        assert len(headers) == 2, "Should handle italic headers"
        assert headers[0]["header"] == "* # Italic Main Header"
        assert headers[1]["header"] == "* ## Italic Sub Header"
        assert headers[1]["parent_header"] == "* # Italic Main Header"

    def test_no_content_headers(self):
        md_text = """
# Header 1
## Header 2
"""
        headers = get_md_header_contents(md_text)
        assert len(headers) == 2, "Should handle headers with no content"
        assert headers[0]["content"] == "", "Header 1 should have empty content"
        assert headers[1]["content"] == "", "Header 2 should have empty content"
        assert headers[1]["parent_header"] == "# Header 1"

    def test_nested_headers_without_content(self):
        md_text = """
# Level 1
## Level 2
### Level 3
"""
        headers = get_md_header_contents(md_text)
        assert len(headers) == 3, "Should handle nested headers without content"
        assert headers[0]["header_level"] == 1
        assert headers[1]["header_level"] == 2
        assert headers[2]["header_level"] == 3
        assert headers[1]["parent_header"] == "# Level 1"
        assert headers[2]["parent_header"] == "## Level 2"


if __name__ == "__main__":
    import pytest
    pytest.main(["-v", __file__])
