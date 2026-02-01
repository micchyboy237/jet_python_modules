# test_md_utils.py

import pytest
from jet.code.markdown_utils.markdown_it_utils import (
    ParseResult,
    parse_and_render_markdown,
)
from markdown_it import MarkdownIt
from mdit_py_plugins.admon import admon_plugin
from mdit_py_plugins.container import container_plugin
from mdit_py_plugins.deflist import deflist_plugin
from mdit_py_plugins.footnote import footnote_plugin
from mdit_py_plugins.tasklists import tasklists_plugin


@pytest.fixture
def basic_md() -> MarkdownIt:
    return MarkdownIt("commonmark")


@pytest.fixture
def tasklist_md() -> MarkdownIt:
    md = MarkdownIt("commonmark")
    md.use(tasklists_plugin, enabled=True)
    return md


# ────────────────────────────────────────────────
# Given-When-Then style tests
# ────────────────────────────────────────────────


def test_parse_simple_paragraph(basic_md: MarkdownIt):
    # Given
    markdown = """
    Hello **world**!
    This is a test.
    """

    # When
    result: ParseResult = parse_and_render_markdown(basic_md, markdown)

    # Then
    assert result["markdown_input"].startswith("Hello **world**!")
    assert "<strong>world</strong>" in result["html"]
    assert len(result["tokens"]) >= 5  # paragraph + strong + text nodes
    assert "env" not in result
    assert "options" in result


def test_tasklist_plugin_is_captured(tasklist_md: MarkdownIt):
    # Given
    markdown = """
    - [ ] todo
    - [x] done
    """

    # When
    result = parse_and_render_markdown(tasklist_md, markdown)

    # Then
    assert "[ ] todo" in result["markdown_input"]
    assert 'type="checkbox"' in result["html"]
    assert "checked" in result["html"]
    assert "env" in result
    # Optional: check that env contains task list data if plugin exposes it


def test_normalization_makes_tests_stable(basic_md: MarkdownIt):
    # Given
    dirty = "   \n\nHello   \n\nworld  \n  "
    clean = "Hello\n\nworld"

    # When
    result_dirty = parse_and_render_markdown(
        basic_md, dirty, normalize_whitespace=False
    )
    result_clean = parse_and_render_markdown(basic_md, dirty, normalize_whitespace=True)

    # Then
    assert result_dirty["markdown_input"] != clean
    assert result_clean["markdown_input"] == clean
    assert result_clean["html"] == result_dirty["html"]  # content same after norm


def test_empty_input_handling(basic_md: MarkdownIt):
    # Given
    empty_cases = ["", "   ", "\n\n\n"]

    for case in empty_cases:
        # When
        result = parse_and_render_markdown(basic_md, case)

        # Then
        assert result["markdown_input"] == ""
        assert result["html"] in ("", "<p></p>", "<p></p>\n")
        assert result["tokens"] == []


@pytest.fixture
def rich_md() -> MarkdownIt:
    """MarkdownIt with many common plugins enabled"""
    md = MarkdownIt("commonmark")
    md.use(tasklists_plugin)
    md.use(deflist_plugin)
    md.use(footnote_plugin)
    md.use(admon_plugin)
    # container_plugin needs configuration - simple example
    md.use(container_plugin, name="spoiler")
    return md


# ────────────────────────────────────────────────
# More realistic / demanding test cases
# ────────────────────────────────────────────────


def test_nested_lists(rich_md: MarkdownIt):
    # Given
    md_text = """
- Top level
  - Sub level 1
    - Deep level
  - Sub level 2

- Another top
    """

    # When
    result: ParseResult = parse_and_render_markdown(rich_md, md_text)

    # Then
    assert "<ul>" in result["html"]
    assert "<li>Deep level</li>" in result["html"]
    assert result["html"].count("<ul>") >= 2  # nested ul


def test_task_list_with_paragraph(rich_md: MarkdownIt):
    # Given
    md_text = """
- [x] Done task

  With a paragraph below it.
  And another line.
    """

    # When
    result = parse_and_render_markdown(rich_md, md_text)

    # Then
    assert 'type="checkbox" checked' in result["html"]
    assert "<p>With a paragraph below it." in result["html"]
    assert "<p>And another line." in result["html"]


def test_footnote(rich_md: MarkdownIt):
    # Given
    md_text = """
Here is some text[^note].

[^note]: And here is the footnote definition.
    """

    # When
    result = parse_and_render_markdown(rich_md, md_text)

    # Then
    assert '<sup class="footnote-ref">' in result["html"]
    assert '<section class="footnotes">' in result["html"]
    assert "And here is the footnote definition." in result["html"]


def test_admonition(rich_md: MarkdownIt):
    # Given
    md_text = """
!!! warning "Careful!"
    This is important.
    """

    # When
    result = parse_and_render_markdown(rich_md, md_text)

    # Then
    assert '<div class="admonition warning">' in result["html"]
    assert "<p>This is important.</p>" in result["html"]


def test_list_with_fenced_code(rich_md: MarkdownIt):
    # Given
    md_text = """
- Item one

  ```python
  print("hello")
  def f():
      return 42
  ```
    """

    # When
    result = parse_and_render_markdown(rich_md, md_text)

    # Then
    assert "<pre" in result["html"]
    assert "<code" in result["html"]
    assert "print(&quot;hello&quot;)" in result["html"]
    assert "return 42" in result["html"]


@pytest.mark.xfail(reason="markdown-it-py table plugin + list nesting is weak")
def test_table_inside_list_item(rich_md: MarkdownIt):
    # Given
    md_text = """
- List item containing table:

  | a | b |
  |---|---|
  | 1 | 2 |
    """

    # When
    result = parse_and_render_markdown(rich_md, md_text)

    # Then — often fails / flattens
    assert "<table>" in result["html"]
    # Many implementations will fail this test → hence xfail
