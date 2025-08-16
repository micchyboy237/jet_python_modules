import pytest
from typing import List

from jet.wordnet.words import split_words, get_words


@pytest.fixture
def setup_nltk(monkeypatch):
    """Fixture to mock NLTK sent_tokenize for consistent testing."""
    def mock_sent_tokenize(text: str) -> List[str]:
        # Simple sentence splitting for testing purposes
        return [s.strip() for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()]
    monkeypatch.setattr("nltk.sent_tokenize", mock_sent_tokenize)
    return mock_sent_tokenize


class TestSplitWords:
    """Tests for the split_words function."""

    def test_given_simple_sentence_when_splitting_words_then_returns_correct_words(self):
        # Given
        input_text = "Hello world"
        expected = ["Hello", "world"]

        # When
        result = split_words(input_text)

        # Then
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_given_text_with_punctuation_when_splitting_words_then_handles_special_chars(self):
        # Given
        input_text = "A.F.&A.M. isn't complex"
        expected = ["A.F.&A.M", "isn't", "complex"]

        # When
        result = split_words(input_text)

        # Then
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_given_empty_string_when_splitting_words_then_returns_empty_list(self):
        # Given
        input_text = ""
        expected = []

        # When
        result = split_words(input_text)

        # Then
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_given_hyphenated_words_when_splitting_words_then_keeps_hyphens(self):
        # Given
        input_text = "well-known author"
        expected = ["well-known", "author"]

        # When
        result = split_words(input_text)

        # Then
        assert result == expected, f"Expected {expected}, but got {result}"


class TestGetWords:
    """Tests for the get_words function."""

    def test_given_single_string_when_getting_words_with_n1_then_returns_single_words(self, setup_nltk):
        # Given
        input_text = "Hello world. This is a test."
        expected = ["Hello", "world", "This", "is", "a", "test"]

        # When
        result = get_words(input_text, n=1)

        # Then
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_given_string_list_when_getting_words_with_n2_then_returns_bigrams(self, setup_nltk):
        # Given
        input_text = ["Hello world.", "This is a test."]
        expected = [
            ["Hello world"],
            ["This is", "is a", "a test"]
        ]

        # When
        result = get_words(input_text, n=2)

        # Then
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_given_string_with_filter_when_getting_words_then_filters_correctly(self, setup_nltk):
        # Given
        input_text = "The quick brown fox"
        def filter_word(w): return len(w) > 3
        expected = ["quick", "brown"]

        # When
        result = get_words(input_text, n=1, filter_word=filter_word)

        # Then
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_given_string_with_ignore_punctuation_when_getting_words_then_removes_punctuation(self, setup_nltk):
        # Given
        input_text = "Hello, world! This is a test."
        expected = ["Hello", "world", "This", "is", "a", "test"]

        # When
        result = get_words(input_text, n=1, ignore_punctuation=True)

        # Then
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_given_invalid_input_when_getting_words_then_raises_value_error(self, setup_nltk):
        # Given
        input_text = 123
        expected_error = "Input must be a string or list of strings"

        # When / Then
        with pytest.raises(ValueError, match=expected_error):
            get_words(input_text)

    def test_given_empty_string_when_getting_words_then_returns_empty_list(self, setup_nltk):
        # Given
        input_text = ""
        expected = []

        # When
        result = get_words(input_text, n=1)

        # Then
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_given_single_word_with_n2_when_getting_words_then_returns_empty_list(self, setup_nltk):
        # Given
        input_text = "Hello"
        expected = []

        # When
        result = get_words(input_text, n=2)

        # Then
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_given_text_with_slash_or_pipe_when_splitting_words_then_splits_on_special_chars(self):
        # Given
        input_text = "save/load yes|no input/output true|false"
        expected = ["save", "load", "yes", "no",
                    "input", "output", "true", "false"]

        # When
        result = split_words(input_text)

        # Then
        assert result == expected, f"Expected {expected}, but got {result}"
