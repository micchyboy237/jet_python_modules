import pytest
from jet.wordnet.analyzers.text_analysis import remove_hyphens


class TestRemoveHyphens:
    """Test suite for remove_hyphens function"""

    def test_replaces_hyphens_with_space(self):
        """Test replacing hyphens between words with a space"""
        # Given
        input_text = "well-known fact-checking"
        expected = "well known fact checking"

        # When
        result = remove_hyphens(input_text)

        # Then
        assert result == expected

    def test_preserves_hyphens_with_spaces(self):
        """Test preserving hyphens when spaces exist around them"""
        # Given
        input_text = "well - known fact - checking"
        expected = "well - known fact - checking"

        # When
        result = remove_hyphens(input_text)

        # Then
        assert result == expected

    def test_handles_empty_string(self):
        """Test handling of empty string input"""
        # Given
        input_text = ""
        expected = ""

        # When
        result = remove_hyphens(input_text)

        # Then
        assert result == expected

    def test_handles_single_word_with_hyphen(self):
        """Test single word with hyphen replaced by space"""
        # Given
        input_text = "self-contained"
        expected = "self contained"

        # When
        result = remove_hyphens(input_text)

        # Then
        assert result == expected

    def test_handles_multiple_hyphens(self):
        """Test word with multiple hyphens replaced by spaces"""
        # Given
        input_text = "state-of-the-art"
        expected = "state of the art"

        # When
        result = remove_hyphens(input_text)

        # Then
        assert result == expected

    def test_handles_mixed_cases(self):
        """Test mixed cases with hyphens and spaces"""
        # Given
        input_text = "high-quality - low-cost product-line"
        expected = "high quality - low cost product line"

        # When
        result = remove_hyphens(input_text)

        # Then
        assert result == expected
