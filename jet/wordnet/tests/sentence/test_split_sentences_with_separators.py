import pytest
from typing import List, Tuple
from jet.wordnet.sentence import split_sentences_with_separators
from nltk.tokenize import sent_tokenize


class TestSplitSentencesWithSeparators:
    def test_single_sentence_without_space_separator(self):
        """Test splitting a single sentence without a space separator."""
        # Given
        input_text = "This is a test."
        expected = ["This is a test."]

        # When
        result = split_sentences_with_separators(input_text)

        # Then
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_multiple_sentences_with_mixed_separators(self):
        """Test splitting multiple sentences separated by newlines and white spaces."""
        # Given
        input_text = "First sentence.\nSecond sentence. Third sentence."
        expected = [
            "First sentence.\n",
            "Second sentence. ",
            "Third sentence."
        ]

        # When
        result = split_sentences_with_separators(input_text)

        # Then
        assert result == expected, f"Expected {expected}, but got {result}"
