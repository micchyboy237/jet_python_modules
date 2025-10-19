
from jet.code.html_utils import is_html

class TestIsHtml:
    """Test suite for is_html function."""

    def test_empty_string(self):
        """Test handling of empty string input."""
        # Given: An empty string
        input_text = ""
        # When: Checking if it's HTML
        result = is_html(input_text)
        # Then: Should return False and log empty input
        expected = False
        assert result == expected

    def test_whitespace_only(self):
        """Test handling of whitespace-only string."""
        # Given: A string with only whitespace
        input_text = "   \n\t  "
        # When: Checking if it's HTML
        result = is_html(input_text)
        # Then: Should return False and log empty input
        expected = False
        assert result == expected

    def test_plain_text(self):
        """Test handling of plain text without HTML."""
        # Given: A plain text string
        input_text = "This is a simple text without any HTML tags."
        # When: Checking if it's HTML
        result = is_html(input_text)
        # Then: Should return False and log no HTML detected
        expected = False
        assert result == expected

    def test_basic_html_tag(self):
        """Test detection of a basic HTML tag."""
        # Given: A string with a simple HTML tag
        input_text = "<p>Hello, world!</p>"
        # When: Checking if it's HTML
        result = is_html(input_text)
        # Then: Should return True and log HTML tags detected
        expected = True
        assert result == expected

    def test_incomplete_html_tag(self):
        """Test detection of incomplete HTML tags."""
        # Given: A string with an incomplete HTML tag
        input_text = "<div>Hello"
        # When: Checking if it's HTML
        result = is_html(input_text)
        # Then: Should return True and log pattern found
        expected = True
        assert result == expected

    def test_html_doctype(self):
        """Test detection of HTML DOCTYPE declaration."""
        # Given: A string with DOCTYPE declaration
        input_text = "<!DOCTYPE html><html><body>Hello</body></html>"
        # When: Checking if it's HTML
        result = is_html(input_text)
        # Then: Should return True and log DOCTYPE pattern
        expected = True
        assert result == expected

    def test_html_comment(self):
        """Test detection of HTML comments."""
        # Given: A string with an HTML comment
        input_text = "<!-- This is a comment -->"
        # When: Checking if it's HTML
        result = is_html(input_text)
        # Then: Should return True and log comment pattern
        expected = True
        assert result == expected

    def test_malformed_html(self):
        """Test detection of malformed HTML content."""
        # Given: A string with malformed HTML
        input_text = "<p>Unclosed tag <div>Still HTML</p>"
        # When: Checking if it's HTML
        result = is_html(input_text)
        # Then: Should return True and log tags detected
        expected = True
        assert result == expected

    def test_markdown_content(self):
        """Test handling of Markdown content."""
        # Given: A string with Markdown syntax
        input_text = "# Heading\n* List item\n**Bold text**"
        # When: Checking if it's HTML
        result = is_html(input_text)
        # Then: Should return False and log no HTML detected
        expected = False
        assert result == expected
