import pytest
from jet.code.html_utils import is_html, valid_html, remove_html_comments


class TestIsHtml:
    @pytest.mark.parametrize(
        "input_text, expected",
        [
            # Given: A complete HTML document
            # When: Checking if it is HTML
            # Then: It should return True
            (
                "<!DOCTYPE html><html><body><p>Hello</p></body></html>",
                True
            ),
            # Given: An HTML fragment with tags
            # When: Checking if it is HTML
            # Then: It should return True
            (
                "<div><span>Test</span></div>",
                True
            ),
            # Given: An HTML comment
            # When: Checking if it is HTML
            # Then: It should return True
            (
                "<!-- This is a comment -->",
                True
            ),
            # Given: Malformed HTML with tags
            # When: Checking if it is HTML
            # Then: It should return True
            (
                "<div>Unclosed tag",
                True
            ),
            # Given: Plain text
            # When: Checking if it is HTML
            # Then: It should return False
            (
                "This is plain text",
                False
            ),
            # Given: Markdown content
            # When: Checking if it is HTML
            # Then: It should return False
            (
                "# Heading\n**Bold** text",
                False
            ),
            # Given: Empty string
            # When: Checking if it is HTML
            # Then: It should return False
            (
                "",
                False
            ),
            # Given: Whitespace-only string
            # When: Checking if it is HTML
            # Then: It should return False
            (
                "   ",
                False
            ),
        ]
    )
    def test_is_html(self, input_text: str, expected: bool) -> None:
        # Given: An input string and expected result
        # When: Calling is_html with the input
        result = is_html(input_text)
        # Then: The result should match the expected value
        assert result == expected, f"Expected {expected} for input '{input_text}', got {result}"


class TestValidHTML:
    @pytest.mark.parametrize(
        "input_text, expected_result",
        [
            # Valid HTML cases
            (
                '<html><head><title>Test</title></head><body><p>Hello</p></body></html>',
                True,
            ),
            (
                '<div class="container">Simple content</div>',
                True,
            ),
            (
                '<span class="badge">New</span>',
                True,
            ),
            # Invalid HTML cases
            (
                '<div>Unclosed tag',
                False,
            ),
            (
                '<p>Mismatched</div>',
                False,
            ),
            (
                '',  # Empty string
                False,
            ),
            # DOCTYPE cases
            (
                '<!DOCTYPE html><html><head><title>Test</title></head><body><p>Hello</p></body></html>',
                True,
            ),
            (
                '<!DOCTYPE html><div>Content</div>',
                True,  # Treated as a fragment
            ),
            (
                '<!DOCTYPE html># Markdown Header',
                False,  # DOCTYPE with Markdown
            ),
        ],
        ids=[
            "valid_full_html",
            "valid_div",
            "valid_span",
            "invalid_unclosed_tag",
            "invalid_mismatched_tag",
            "invalid_empty_string",
            "valid_doctype_full_html",
            "valid_doctype_fragment",
            "invalid_doctype_markdown",
        ],
    )
    def test_html_syntax(self, input_text: str, expected_result: bool):
        # Given: An input string to validate as HTML
        # When: The valid_html function is called
        result = valid_html(input_text)
        # Then: The result should match the expected outcome
        assert result == expected_result, f"Expected {expected_result} for input '{input_text}', got {result}"

    @pytest.mark.parametrize(
        "input_text, expected_result",
        [
            (
                '# Project Overview\nWelcome to our **project**!',
                False,
            ),
            (
                '```python\ndef greet(name: str) -> str:\n    return f"Hello, {name}!"\n```',
                False,
            ),
            (
                '- List item 1\n    - Nested item\n- List item 2',
                False,
            ),
            (
                '| Endpoint | Method | Description |\n|----------|--------|-------------|\n| /api/users | GET | Fetch users |',
                False,
            ),
            (
                '[Link](https://project.com)',
                False,
            ),
            (
                '![Image](https://project.com/logo.png)',
                False,
            ),
            (
                '> **Note**: Check the [docs](https://docs.project.com).',
                False,
            ),
            (
                '[^1]: Footnote definition here.',
                False,
            ),
        ],
        ids=[
            "markdown_heading",
            "markdown_code_block",
            "markdown_list",
            "markdown_table",
            "markdown_link",
            "markdown_image",
            "markdown_blockquote",
            "markdown_footnote",
        ],
    )
    def test_markdown_syntax(self, input_text: str, expected_result: bool):
        # Given: An input string containing Markdown syntax
        # When: The valid_html function is called
        result = valid_html(input_text)
        # Then: The result should be False as Markdown is not valid HTML
        assert result == expected_result, f"Expected {expected_result} for Markdown input '{input_text}', got {result}"

    def test_mixed_html_and_markdown(self):
        # Given: An input string with both HTML and Markdown
        input_text = '<div class="alert">This is HTML</div>\n# Markdown Header\n**Bold text**'
        expected_result = False
        # When: The valid_html function is called
        result = valid_html(input_text)
        # Then: The result should be False due to Markdown content
        assert result == expected_result, f"Expected {expected_result} for mixed input, got {result}"

    @pytest.mark.parametrize(
        "input_text, expected_result",
        [
            (
                '<?xml version="1.0"?><root>XML content</root>',
                False,
            ),
            (
                'Plain text without tags',
                False,
            ),
            (
                '<![CDATA[Some content]]>',
                False,
            ),
        ],
        ids=[
            "xml_content",
            "plain_text",
            "cdata_section",
        ],
    )
    def test_non_html_content(self, input_text: str, expected_result: bool):
        # Given: An input string with non-HTML content
        # When: The valid_html function is called
        result = valid_html(input_text)
        # Then: The result should be False as the content is not valid HTML
        assert result == expected_result, f"Expected {expected_result} for non-HTML input '{input_text}', got {result}"


class TestRemoveHtmlComments:
    @pytest.mark.parametrize(
        "input_text, expected",
        [
            (
                '< Caspio Script --><div>Hello<!-- comment -->World</div>',
                '<div>HelloWorld</div>',
            ),
            (
                """<div>
<!-- This is
a multiline
comment -->
<p>Content</p>
</div>""",
                """<div>
<p>Content</p>
</div>""",
            ),
            (
                '<!--1--><div><!--2--></div><!--3-->',
                '<div></div>',
            ),
            (
                '<p>No comments here</p>',
                '<p>No comments here</p>',
            ),
            (
                '<div><!-- outer <!-- inner --> still outer --></div>',
                '<div> still outer </div>',
            ),
            (
                '',  # Empty string
                '',
            ),
        ],
        ids=[
            "single_line_comment",
            "multiline_comment",
            "multiple_comments",
            "no_comments",
            "nested_like_content",
            "empty_string",
        ],
    )
    def test_remove_comments(self, input_text: str, expected: str):
        # Given: An input string with or without HTML comments
        # When: The remove_html_comments function is called
        result = remove_html_comments(input_text)
        # Then: The result should match the expected output
        assert result == expected, f"Expected '{expected}', got '{result}'"
