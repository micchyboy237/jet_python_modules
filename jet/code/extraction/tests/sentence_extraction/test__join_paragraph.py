from jet.code.extraction.sentence_extraction import _join_paragraph

class TestJoinParagraph:
    """BDD-style tests for _join_paragraph using real-world document examples."""

    # Given: a single string (possibly with extra whitespace)
    # When: calling _join_paragraph
    # Then: returns stripped version
    def test_single_string_input(self):
        result = _join_paragraph("  Hello world  \n")
        expected = "Hello world"
        assert result == expected

    # Given: empty string
    # When: calling _join_paragraph
    # Then: returns empty string
    def test_empty_string(self):
        result = _join_paragraph("")
        expected = ""
        assert result == expected

    # Given: list of clean strings
    # When: calling _join_paragraph
    # Then: joins with single newlines, no extra lines
    def test_list_of_clean_strings(self):
        result = _join_paragraph(["First line", "Second line"])
        expected = "First line\nSecond line"
        assert result == expected

    # Given: list with whitespace and empty strings
    # When: calling _join_paragraph
    # Then: trims and skips empty entries
    def test_list_with_whitespace_and_empty(self):
        result = _join_paragraph(["  Alpha  ", "", "  Beta  ", " \t "])
        expected = "Alpha\nBeta"
        assert result == expected

    # Given: nested list with mixed content
    # When: calling _join_paragraph
    # Then: flattens, trims, filters empties, joins correctly
    def test_nested_list_mixed_content(self):
        result = _join_paragraph([
            "Title",
            ["  Subtitle  ", ["Inner"]],
            None,
            ["  ", "Final line  "]
        ])
        expected = "Title\nSubtitle\nInner\nFinal line"
        assert result == expected

    # Given: deeply nested with only whitespace
    # When: calling _join_paragraph
    # Then: results in empty string (all filtered out)
    def test_deeply_nested_only_whitespace(self):
        result = _join_paragraph([["   ", [" \n "]], " \t "])
        expected = ""
        assert result == expected
