import pytest
from typing import List

from jet.wordnet.words import get_words


@pytest.fixture
def setup_nltk(monkeypatch):
    """Fixture to mock NLTK sent_tokenize for consistent testing."""
    def mock_sent_tokenize(text: str) -> List[str]:
        # Simple sentence splitting for testing purposes
        return [s.strip() for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()]
    monkeypatch.setattr("nltk.sent_tokenize", mock_sent_tokenize)
    return mock_sent_tokenize


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

    def test_given_text_with_delimiters_when_getting_words_then_splits_on_special_chars(self, setup_nltk):
        # Given
        input_text = "input/output_yes:no. This;is true,false."
        expected_n1 = ["input", "output", "yes",
                       "no", "This", "is", "true", "false"]
        expected_n2 = ["input output", "output yes",
                       "yes no", "This is", "is true", "true false"]
        input_list = ["input/output_yes:no;", "true,false_test."]
        expected_list_n1 = [
            ["input", "output", "yes", "no"],
            ["true", "false", "test"]
        ]

        # When
        result_n1 = get_words(input_text, n=1)
        result_n2 = get_words(input_text, n=2)
        result_list_n1 = get_words(input_list, n=1)

        # Then
        assert result_n1 == expected_n1, f"Expected {expected_n1}, but got {result_n1}"
        assert result_n2 == expected_n2, f"Expected {expected_n2}, but got {result_n2}"
        assert result_list_n1 == expected_list_n1, f"Expected {expected_list_n1}, but got {result_list_n1}"
