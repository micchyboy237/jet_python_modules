from jet.code.html_utils import dl_to_md, convert_dl_blocks_to_md
import re


class TestDlToMd:
    """Behavior-driven tests for <dl> → Markdown conversion."""

    def setup_method(self):
        self.pattern = re.compile(r"<dl[^>]*>\s*(.*?)\s*</dl>", re.DOTALL | re.IGNORECASE)

    def test_single_term_single_definition(self):
        # Given a simple dl with one dt/dd pair
        html = "<dl><dt>Release Date</dt><dd>January 1, 2025</dd></dl>"
        match = self.pattern.search(html)

        # When dl_to_md is called
        result = dl_to_md(match)
        expected = "Release Date\n: January 1, 2025\n\n"

        # Then it should produce valid Markdown
        assert result == expected

    def test_single_term_multiple_definitions(self):
        # Given a term with two dd entries
        html = "<dl><dt>Platforms</dt><dd>PC</dd><dd>Mac</dd></dl>"
        match = self.pattern.search(html)

        # When dl_to_md is called
        result = dl_to_md(match)
        expected = "Platforms\n: PC\n: Mac\n\n"

        assert result == expected

    def test_multiple_terms(self):
        # Given multiple dt/dd pairs
        html = """
        <dl>
          <dt>Release Date</dt><dd>January 1, 2025</dd>
          <dt>Platforms</dt><dd>PC</dd><dd>Mac</dd>
        </dl>
        """
        match = self.pattern.search(html)

        result = dl_to_md(match)
        expected = (
            "Release Date\n: January 1, 2025\n\n"
            "Platforms\n: PC\n: Mac\n\n"
        )

        assert result == expected

    def test_orphan_dd(self):
        # Given a <dd> without a preceding <dt>
        html = "<dl><dd>Orphaned definition</dd></dl>"
        match = self.pattern.search(html)

        result = dl_to_md(match)
        expected = ": Orphaned definition\n\n"
        assert result == expected

    def test_convert_dl_blocks_to_md_wrapper(self):
        # Given full HTML containing a dl block
        html = "<p>Intro</p><dl><dt>Term</dt><dd>Definition</dd></dl><p>Outro</p>"
        result = convert_dl_blocks_to_md(html)
        expected = "<p>Intro</p><pre class=\"jet-dl-block\">Term\n: Definition\n\n</pre><p>Outro</p>"
        assert expected in result


class TestConvertDlBlocksToMdPreserveNewlines:
    """Test that converted Markdown preserves original newlines and integrates correctly with surrounding HTML."""

    # Given: HTML with <dl> containing nested elements and surrounding content with newlines
    # When: convert_dl_blocks_to_md is called
    # Then: The <dl> is replaced with Markdown inside a <pre> wrapper that preserves newlines
    def test_nested_elements_and_surrounding_newlines(self):
        html = """
        <div class="intro">
            <p>Game overview:</p>
        </div>

        <dl>
            <dt>Genre</dt>
            <dd><strong>Action</strong> Adventure</dd>
            <dt>Developer</dt>
            <dd><a href="/dev">Studio X</a></dd>
        </dl>

        <p>Available now!</p>
        """
        result = convert_dl_blocks_to_md(html)
        expected = """
<div class="intro">
<p>Game overview:</p>
</div>

<pre class="jet-dl-block">Genre
: Action Adventure

Developer
: Studio X

</pre>

<p>Available now!</p>
""".strip()
        assert result == expected

    # Given: Multiple <dl> blocks with complex content
    # When: convert_dl_blocks_to_md processes all
    # Then: Each is wrapped in <pre> independently
    def test_multiple_dl_blocks(self):
        html = """
        <dl><dt>A</dt><dd>1</dd></dl>
        <p>---</p>
        <dl><dt>B</dt><dd>2</dd><dd>3</dd></dl>
        """
        result = convert_dl_blocks_to_md(html)
        expected = """
<pre class="jet-dl-block">A
: 1

</pre>
<p>---</p>
<pre class="jet-dl-block">B
: 2
: 3

</pre>
""".strip()
        assert result == expected
