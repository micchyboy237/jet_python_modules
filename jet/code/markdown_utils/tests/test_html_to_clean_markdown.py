# tests/test_html_to_clean_markdown.py
"""
Unit tests for html_to_clean_markdown() wrapper around markdownify.
"""

import pytest
from jet.code.markdown_utils.markdown_it_utils import html_to_clean_markdown

# -------------------------------------------------------------------------
# Basic inline formatting
# -------------------------------------------------------------------------


def test_converts_strong_em_code_inline():
    # Given HTML with common inline markup
    html = """
    <p>This is <strong>bold</strong>, <em>italic</em>, and <code>code</code>.</p>
    """

    # When converted with default settings
    result = html_to_clean_markdown(html)

    # Then markdown contains expected inline syntax
    expected = "This is **bold**, *italic*, and `code`."
    assert expected in result


def test_respects_strong_em_symbol_parameter():
    # Given HTML with bold & italic
    html = "<p><strong>bold</strong> and <em>italic</em></p>"

    # When using underscore for emphasis
    result = html_to_clean_markdown(html, strong_em_symbol="_")

    # Then output uses underscores
    expected = "__bold__ and _italic_"
    assert expected in result


# -------------------------------------------------------------------------
# Headings
# -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "heading_style, expected_prefix",
    [
        ("ATX", "# "),
        ("ATX_CLOSED", "# "),
        ("underlined", "Main Title\n=========="),
    ],
)
def test_heading_style_parameter(heading_style, expected_prefix):
    # Given simple heading HTML
    html = "<h1>Main Title</h1><p>Content.</p>"

    # When converted with different heading styles
    result = html_to_clean_markdown(html, heading_style=heading_style)

    # Then heading is formatted correctly
    assert result.strip().startswith(expected_prefix)


# -------------------------------------------------------------------------
# Lists – unordered & nested
# -------------------------------------------------------------------------


def test_unordered_list_default_bullets():
    # Given nested unordered list
    html = """
    <ul>
      <li>Top level
        <ul>
          <li>Second level</li>
          <li>Another</li>
        </ul>
      </li>
      <li>Second top</li>
    </ul>
    """

    # When using default bullets
    result = html_to_clean_markdown(html)

    # Then uses - for top level, * for nested (common markdownify default)
    expected_lines = ["- Top level", "  * Second level", "  * Another", "- Second top"]
    for line in expected_lines:
        assert line in result


def test_custom_bullets_parameter():
    # Given same nested list
    html = """
    <ul><li>A</li><li>B<ul><li>C</li></ul></li></ul>
    """

    # When using custom bullet sequence
    result = html_to_clean_markdown(html, bullets="+*-")

    # Then uses + → * → -
    assert result.strip().startswith("+ A")
    assert "  * B" in result
    assert "    - C" in result


def test_ordered_list():
    # Given ordered list HTML
    html = """
    <ol>
      <li>First</li>
      <li>Second</li>
      <li value="5">Fifth</li>
    </ol>
    """

    # When converted
    result = html_to_clean_markdown(html)

    # Then produces numbered list (markdownify usually starts from 1 unless value=)
    expected = "1. First\n2. Second\n5. Fifth"
    assert expected in result.strip()


# -------------------------------------------------------------------------
# Tables
# -------------------------------------------------------------------------


def test_basic_table_without_header_inference():
    # Given simple table
    html = """
    <table>
      <tr><td>A</td><td>B</td></tr>
      <tr><td>1</td><td>2</td></tr>
    </table>
    """

    # When header inference is off
    result = html_to_clean_markdown(html, table_infer_header=False)

    # Then no header separator (markdownify default behavior)
    assert "|" in result
    assert "---" not in result


def test_table_with_header_inference():
    # Given table with first row as header-like
    html = """
    <table>
      <tr><th>Name</th><th>Age</th></tr>
      <tr><td>Alice</td><td>30</td></tr>
    </table>
    """

    # When inference enabled
    result = html_to_clean_markdown(html, table_infer_header=True)

    # Then header separator appears
    assert "| Name | Age |" in result
    assert "| --- | --- |" in result


# -------------------------------------------------------------------------
# Code blocks & pre
# -------------------------------------------------------------------------


def test_fenced_code_block():
    # Given <pre><code> block
    html = """
    <pre><code class="language-python">print("Hello")\nx = 42</code></pre>
    """

    # When converted
    result = html_to_clean_markdown(html)

    # Then produces fenced code block
    expected = '```python\nprint("Hello")\nx = 42\n```'
    assert expected in result


# -------------------------------------------------------------------------
# Stripping unwanted content
# -------------------------------------------------------------------------


def test_strip_parameter_removes_script_and_style():
    # Given HTML with unwanted tags
    html = """
    <div>
      <p>Content</p>
      <script>alert('xss');</script>
      <style>body {color:red}</style>
    </div>
    """

    # When stripping script & style
    result = html_to_clean_markdown(html, strip=["script", "style"])

    # Then script & style content is gone
    assert "alert" not in result
    assert "color:red" not in result
    assert "Content" in result


# -------------------------------------------------------------------------
# Autolinks
# -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "autolink_enabled, expected_contains_angle",
    [
        (True, True),
        (False, False),
    ],
)
def test_autolink_parameter(autolink_enabled, expected_contains_angle):
    # Given paragraph with plain URL
    html = "<p>See https://example.com for details.</p>"

    # When autolink is toggled
    result = html_to_clean_markdown(html, autolink=autolink_enabled)

    # Then plain URL is converted to <> only when enabled
    if expected_contains_angle:
        assert "<https://example.com>" in result
    else:
        assert "<https" not in result


# -------------------------------------------------------------------------
# Edge cases
# -------------------------------------------------------------------------


def test_empty_input():
    # Given empty or whitespace-only HTML
    html = "   \n\t   "

    # When converted
    result = html_to_clean_markdown(html)

    # Then result is empty string
    assert result == ""


def test_malformed_html_is_handled_gracefully():
    # Given broken HTML
    html = "<p>Unclosed <b>tag and <div>weirdness"

    # When converted
    result = html_to_clean_markdown(html)

    # Then still produces usable markdown (BeautifulSoup recovers)
    assert "**tag" in result or "tag" in result
    assert result.strip() != ""
