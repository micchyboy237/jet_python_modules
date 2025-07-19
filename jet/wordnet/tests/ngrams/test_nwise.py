import pytest
from typing import Iterator, Tuple

from jet.wordnet.n_grams import nwise


class TestNwise:
    def test_single_line_sliding_window_n2(self):
        # Given a single line of text with multiple words
        input_text = "the quick brown fox"
        expected = [
            ("the", "quick"),
            ("quick", "brown"),
            ("brown", "fox")
        ]
        # When nwise is called with n=2
        result = list(nwise(input_text.split(), n=2))
        # Then it returns pairs of adjacent words
        assert result == expected

    def test_multi_line_sliding_window_n2(self):
        # Given multiple lines of text
        input_text = "the quick\nbrown fox\njumps high"
        expected = [
            ("the", "quick"),
            ("brown", "fox"),
            ("jumps", "high")
        ]
        # When nwise is called with n=2
        result = list(nwise(input_text, n=2))
        # Then it returns pairs within each line, not across newlines
        assert result == expected

    def test_sliding_window_n3(self):
        # Given a single line with enough words for n=3
        input_text = "the quick brown fox jumps"
        expected = [
            ("the", "quick", "brown"),
            ("quick", "brown", "fox"),
            ("brown", "fox", "jumps")
        ]
        # When nwise is called with n=3
        result = list(nwise(input_text.split(), n=3))
        # Then it returns triplets of adjacent words
        assert result == expected

    def test_line_too_short_for_window(self):
        # Given a line with fewer words than the window size
        input_text = "the\nquick brown"
        expected = [("quick", "brown")]
        # When nwise is called with n=2
        result = list(nwise(input_text, n=2))
        # Then it only returns pairs from lines with enough words
        assert result == expected

    def test_empty_input(self):
        # Given an empty input
        input_text = ""
        expected = []
        # When nwise is called with n=2
        result = list(nwise(input_text, n=2))
        # Then it returns an empty list
        assert result == expected

    def test_invalid_n_zero(self):
        # Given an input with valid text but n=0
        input_text = "the quick brown"
        # When nwise is called with n=0
        result = nwise(input_text.split(), n=0)
        # Then it returns None or an empty iterator
        assert list(result) == []

    def test_single_word_lines(self):
        # Given multiple lines with single words
        input_text = "the\nquick\nbrown"
        expected = []
        # When nwise is called with n=2
        result = list(nwise(input_text, n=2))
        # Then it returns an empty list since no line has enough words
        assert result == expected
