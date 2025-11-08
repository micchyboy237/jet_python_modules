from jet.code.extraction.sentence_extraction import _flatten_list

class TestFlattenList:
    """BDD-style tests for _flatten_list using real-world examples."""

    # Given: flat list of strings
    # When: calling _flatten_list
    # Then: returns the same list
    def test_flat_list_of_strings(self):
        result = _flatten_list(["a", "b", "c"])
        expected = ["a", "b", "c"]
        assert result == expected

    # Given: empty list
    # When: calling _flatten_list
    # Then: returns empty list
    def test_empty_list(self):
        result = _flatten_list([])
        expected = []
        assert result == expected

    # Given: single-level nested list
    # When: calling _flatten_list
    # Then: extracts all strings
    def test_single_level_nesting(self):
        result = _flatten_list(["a", ["b", "c"], "d"])
        expected = ["a", "b", "c", "d"]
        assert result == expected

    # Given: multi-level nested list
    # When: calling _flatten_list
    # Then: recursively flattens to all strings
    def test_multi_level_nesting(self):
        result = _flatten_list(["x", ["y", ["z"]], [["w"]]])
        expected = ["x", "y", "z", "w"]
        assert result == expected

    # Given: mixed types (non-list, non-str elements)
    # When: calling _flatten_list
    # Then: ignores non-string, non-list elements
    def test_ignores_non_string_non_list(self):
        result = _flatten_list(["a", 123, None, {"k": "v"}, ["b"]])
        expected = ["a", "b"]
        assert result == expected
