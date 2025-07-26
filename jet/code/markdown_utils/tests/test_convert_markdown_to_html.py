import pytest
from typing import List
import logging
from jet.code.markdown_utils._converters import convert_markdown_to_html
from jet.code.markdown_types.converter_types import MarkdownExtensions

logger = logging.getLogger(__name__)


class TestMarkdownRenderer:
    def test_extra_extension(self):
        # Given: Markdown content with a code block and table
        md_content = (
            "```python\n"
            "def hello():\n"
            "    print(\"Hello, World!\")\n"
            "```\n"
            "| Header1 | Header2 |\n"
            "|---------|---------|\n"
            "| Cell1   | Cell2   |\n"
        )
        exts: MarkdownExtensions = {"extensions": ["extra"]}
        expected = (
            '<pre><code class="language-python">def hello():\n'
            '    print(&quot;Hello, World!&quot;)\n'
            '</code></pre>\n'
            '<table>\n'
            '<thead>\n'
            '<tr>\n'
            '<th>Header1</th>\n'
            '<th>Header2</th>\n'
            '</tr>\n'
            '</thead>\n'
            '<tbody>\n'
            '<tr>\n'
            '<td>Cell1</td>\n'
            '<td>Cell2</td>\n'
            '</tr>\n'
            '</tbody>\n'
            '</table>'
        )
        # When: Converting Markdown to HTML
        result = convert_markdown_to_html(md_content, exts)
        # Then: The HTML should match the expected output
        assert result.strip() == expected.strip()

    def test_attr_list_extension(self):
        # Given: Markdown content with attributes
        md_content = "A paragraph with custom attributes {#para1 .class1 style=\"color: blue;\"}"
        exts: MarkdownExtensions = {"extensions": ["attr_list"]}
        expected = '<p>A paragraph with custom attributes {#para1 .class1 style="color: blue;"}</p>'
        # When: Converting Markdown to HTML
        result = convert_markdown_to_html(md_content, exts)
        # Then: The HTML should match the expected output
        assert result.strip() == expected.strip()

    def test_def_list_extension(self):
        # Given: Markdown content with a definition list
        md_content = (
            "Term 1\n"
            ": Definition for term 1.\n\n"
            "Term 2\n"
            ": Definition for term 2.\n"
        )
        exts: MarkdownExtensions = {"extensions": ["def_list"]}
        expected = (
            '<dl>\n'
            '<dt>Term 1</dt>\n'
            '<dd>Definition for term 1.</dd>\n'
            '<dt>Term 2</dt>\n'
            '<dd>Definition for term 2.</dd>\n'
            '</dl>'
        )
        # When: Converting Markdown to HTML
        result = convert_markdown_to_html(md_content, exts)
        # Then: The HTML should match the expected output
        assert result.strip() == expected.strip()

    def test_fenced_code_extension(self):
        # Given: Markdown content with a JavaScript code block
        md_content = (
            "```javascript\n"
            "function greet() {\n"
            "    console.log(\"Hello!\");\n"
            "}\n"
            "```\n"
        )
        exts: MarkdownExtensions = {"extensions": ["fenced_code"]}
        expected = (
            '<pre><code class="language-javascript">function greet() {\n'
            '    console.log(&quot;Hello!&quot;);\n'
            '}\n'
            '</code></pre>'
        )
        # When: Converting Markdown to HTML
        result = convert_markdown_to_html(md_content, exts)
        # Then: The HTML should match the expected output
        assert result.strip() == expected.strip()

    def test_footnotes_extension(self):
        # Given: Markdown content with a footnote
        md_content = (
            "Here is some text[^1].\n\n"
            "[^1]: This is a footnote.\n"
        )
        exts: MarkdownExtensions = {"extensions": ["footnotes"]}
        expected = (
            '<p>Here is some text<sup id="fnref:1"><a class="footnote-ref" href="#fn:1">1</a></sup>.</p>\n'
            '<div class="footnote">\n'
            '<hr />\n'
            '<ol>\n'
            '<li id="fn:1">\n'
            '<p>This is a footnote.&#160;<a class="footnote-backref" href="#fnref:1" title="Jump back to footnote 1 in the text">&#8617;</a></p>\n'
            '</li>\n'
            '</ol>\n'
            '</div>'
        )
        # When: Converting Markdown to HTML
        result = convert_markdown_to_html(md_content, exts)
        # Then: The HTML should match the expected output
        assert result.strip() == expected.strip()

    def test_md_in_html_extension(self):
        # Given: Markdown content with Markdown inside HTML
        md_content = (
            '<div markdown="1">\n'
            '*Emphasis* inside a div.\n'
            '</div>\n'
        )
        exts: MarkdownExtensions = {"extensions": ["md_in_html"]}
        expected = '<div>\n<p><em>Emphasis</em> inside a div.</p>\n</div>'
        # When: Converting Markdown to HTML
        result = convert_markdown_to_html(md_content, exts)
        # Then: The HTML should match the expected output
        assert result.strip() == expected.strip()

    def test_multiple_extensions(self):
        # Given: Markdown content with code, table, and footnote
        md_content = (
            "```python\n"
            "def example():\n"
            "    pass\n"
            "```\n"
            "| Col1 | Col2 |\n"
            "|------|------|\n"
            "| A    | B    |\n\n"
            "Text with footnote[^1].\n\n"
            "[^1]: Footnote content.\n"
        )
        exts: MarkdownExtensions = {"extensions": [
            "fenced_code", "tables", "footnotes"]}
        expected = (
            '<pre><code class="language-python">def example():\n'
            '    pass\n'
            '</code></pre>\n'
            '<table>\n'
            '<thead>\n'
            '<tr>\n'
            '<th>Col1</th>\n'
            '<th>Col2</th>\n'
            '</tr>\n'
            '</thead>\n'
            '<tbody>\n'
            '<tr>\n'
            '<td>A</td>\n'
            '<td>B</td>\n'
            '</tr>\n'
            '</tbody>\n'
            '</table>\n'
            '<p>Text with footnote<sup id="fnref:1"><a class="footnote-ref" href="#fn:1">1</a></sup>.</p>\n'
            '<div class="footnote">\n'
            '<hr />\n'
            '<ol>\n'
            '<li id="fn:1">\n'
            '<p>Footnote content.&#160;<a class="footnote-backref" href="#fnref:1" title="Jump back to footnote 1 in the text">&#8617;</a></p>\n'
            '</li>\n'
            '</ol>\n'
            '</div>'
        )
        # When: Converting Markdown to HTML
        result = convert_markdown_to_html(md_content, exts)
        # Then: The HTML should match the expected output
        assert result.strip() == expected.strip()

    def test_abbr_extension(self):
        # Given: Markdown content with proper abbreviation syntax
        logger.debug("Starting test_abbr_extension")
        md_content = (
            "LOL and WTF are abbreviations.\n\n"
            "*[LOL]: Laughing Out Loud\n"
            "*[WTF]: What The Fudge\n"
        )
        exts: MarkdownExtensions = {"extensions": ["abbr"]}
        expected = (
            '<p><abbr title="Laughing Out Loud">LOL</abbr> and '
            '<abbr title="What The Fudge">WTF</abbr> are abbreviations.</p>'
        )
        # When: Converting markdown to HTML
        logger.debug(f"Input markdown: {md_content}")
        result = convert_markdown_to_html(md_content, exts)
        logger.debug(f"Rendered HTML: {result}")
        # Then: The result matches the expected HTML with abbreviations
        assert result.strip() == expected.strip()

    def test_codehilite_extension(self):
        # Given: Markdown content with code block and fenced_code extension
        logger.debug("Starting test_codehilite_extension")
        md_content = (
            "```python\n"
            "def example():\n"
            "    print('Hello')\n"
            "```\n"
        )
        exts: MarkdownExtensions = {
            "extensions": ["codehilite", "fenced_code"]}
        expected = (
            '<div class="codehilite"><pre><span></span><code><span class="k">def</span><span class="w"> </span><span class="nf">example</span><span class="p">():</span>\n'
            '    <span class="nb">print</span><span class="p">(</span><span class="s1">&#39;Hello&#39;</span><span class="p">)</span>\n'
            '</code></pre></div>'
        )
        # When: Converting markdown to HTML
        logger.debug(f"Input markdown: {md_content}")
        result = convert_markdown_to_html(md_content, exts)
        logger.debug(f"Rendered HTML: {result}")
        # Then: The result matches the expected HTML with highlighted code
        assert result.strip() == expected.strip()

    def test_legacy_attrs_extension(self):
        # Given: Markdown content with legacy attribute syntax
        logger.debug("Starting test_legacy_attrs_extension")
        md_content = (
            "Paragraph with *legacy* attributes {id=\"my-id\" class=\"my-class\"}.\n"
        )
        exts: MarkdownExtensions = {"extensions": ["legacy_attrs"]}
        expected = (
            '<p>Paragraph with <em>legacy</em> attributes {id="my-id" class="my-class"}.</p>'
        )
        # When: Converting markdown to HTML
        logger.debug(f"Input markdown: {md_content}")
        result = convert_markdown_to_html(md_content, exts)
        logger.debug(f"Rendered HTML: {result}")
        # Then: The result matches the expected HTML with legacy attributes
        assert result.strip() == expected.strip()

    def test_legacy_em_extension(self):
        # Given: Markdown content with legacy emphasis syntax
        logger.debug("Starting test_legacy_em_extension")
        md_content = (
            "*italic* and **bold** text.\n"
        )
        exts: MarkdownExtensions = {"extensions": ["legacy_em"]}
        expected = (
            '<p><em>italic</em> and <strong>bold</strong> text.</p>'
        )
        # When: Converting markdown to HTML
        logger.debug(f"Input markdown: {md_content}")
        result = convert_markdown_to_html(md_content, exts)
        logger.debug(f"Rendered HTML: {result}")
        # Then: The result matches the expected HTML with legacy emphasis tags
        assert result.strip() == expected.strip()

    def test_sane_lists_extension(self):
        # Given: Markdown content with proper list syntax
        logger.debug("Starting test_sane_lists_extension")
        md_content = (
            "- Item 1\n"
            "    1. Subitem A\n"
            "    2. Subitem B\n"
            "- Item 2\n"
        )
        exts: MarkdownExtensions = {"extensions": ["sane_lists"]}
        expected = (
            '<ul>\n'
            '<li>Item 1<ol>\n'
            '<li>Subitem A</li>\n'
            '<li>Subitem B</li>\n'
            '</ol>\n'
            '</li>\n'
            '<li>Item 2</li>\n'
            '</ul>'
        )
        # When: Converting markdown to HTML
        logger.debug(f"Input markdown: {md_content}")
        result = convert_markdown_to_html(md_content, exts)
        logger.debug(f"Rendered HTML: {result}")
        # Then: The result matches the expected HTML with sane list structure
        assert result.strip() == expected.strip()

    def test_smarty_extension(self):
        # Given: Markdown content with smart typography
        logger.debug("Starting test_smarty_extension")
        md_content = (
            "He said, \"Hello...\" and used -- and --- in text.\n"
        )
        exts: MarkdownExtensions = {"extensions": ["smarty"]}
        expected = (
            '<p>He said, &ldquo;Hello&hellip;&rdquo; and used &ndash; and &mdash; in text.</p>'
        )
        # When: Converting markdown to HTML
        logger.debug(f"Input markdown: {md_content}")
        result = convert_markdown_to_html(md_content, exts)
        logger.debug(f"Rendered HTML: {result}")
        # Then: The result matches the expected HTML with smart typography
        assert result.strip() == expected.strip()

    def test_toc_extension(self):
        # Given: Markdown content with headers and TOC marker
        logger.debug("Starting test_toc_extension")
        md_content = (
            "[TOC]\n\n"
            "# Heading 1\n"
            "## Heading 2\n"
            "### Heading 3\n"
        )
        exts: MarkdownExtensions = {"extensions": ["toc"]}
        expected = (
            '<div class="toc">\n'
            '<ul>\n'
            '<li><a href="#heading-1">Heading 1</a><ul>\n'
            '<li><a href="#heading-2">Heading 2</a><ul>\n'
            '<li><a href="#heading-3">Heading 3</a></li>\n'
            '</ul>\n'
            '</li>\n'
            '</ul>\n'
            '</li>\n'
            '</ul>\n'
            '</div>\n'
            '<h1 id="heading-1">Heading 1</h1>\n'
            '<h2 id="heading-2">Heading 2</h2>\n'
            '<h3 id="heading-3">Heading 3</h3>'
        )
        # When: Converting markdown to HTML
        logger.debug(f"Input markdown: {md_content}")
        result = convert_markdown_to_html(md_content, exts)
        logger.debug(f"Rendered HTML: {result}")
        # Then: The result matches the expected HTML with table of contents
        assert result.strip() == expected.strip()

    def test_wikilinks_extension(self):
        # Given: Markdown content with a single wiki-style link
        logger.debug("Starting test_wikilinks_extension")
        md_content = (
            "This is a [[WikiLink]].\n"
        )
        exts: MarkdownExtensions = {"extensions": ["wikilinks"]}
        expected = (
            '<p>This is a <a class="wikilink" href="/WikiLink/">WikiLink</a>.</p>'
        )
        # When: Converting markdown to HTML
        logger.debug(f"Input markdown: {md_content}")
        result = convert_markdown_to_html(md_content, exts)
        logger.debug(f"Rendered HTML: {result}")
        # Then: The result matches the expected HTML with wiki links
        assert result.strip() == expected.strip()
