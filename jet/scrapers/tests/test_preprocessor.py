import pytest
from jet.scrapers.preprocessor import remove_display_none_elements, clean_html_noise, convert_html_to_markdown, html_to_markdown, normalize_whitespace


def test_remove_display_none_elements():
    """Test removal of display:none elements."""
    input_html = """
    <div>
        <p>Visible content</p>
        <div style="display: none;">Hidden content</div>
    </div>
    """
    expected = "<div> <p>Visible content</p> </div>"
    result = remove_display_none_elements(input_html)
    assert result == expected


def test_clean_html_noise():
    """Test cleaning of noisy HTML elements and boilerplate."""
    input_html = """
    <div>
        <script>alert('test');</script>
        <nav>Navigation</nav>
        <p>Main content</p>
        <footer>All rights reserved</footer>
        <div class="ad-banner">Advertisement</div>
    </div>
    """
    expected = "<div> <p>Main content</p> </div>"
    result = clean_html_noise(input_html)
    assert result == expected


def test_convert_html_to_markdown():
    """Test HTML to Markdown conversion."""
    input_html = """
    <div>
        <h1>Title</h1>
        <p>Paragraph with <a href="test.com">link</a>.</p>
    </div>
    """
    expected = "# Title\n\nParagraph with link."
    result = convert_html_to_markdown(input_html, ignore_links=True)
    assert result == expected


def test_html_to_markdown():
    """Test full HTML to Markdown pipeline with selectors."""
    input_html = """
    <body>
        <h1>Main Title</h1>
        <nav>Navigation</nav>
        <p>Content</p>
    </body>
    """
    expected = "# Main Title\n\nContent"
    result = html_to_markdown(input_html, remove_selectors=["nav"])
    assert result == expected
