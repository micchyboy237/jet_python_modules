import pytest
from jet.wordnet.keywords.spelling_corrector import SpellingCorrector
from typing import List, Dict, Optional


@pytest.fixture
def spelling_corrector():
    """Fixture to initialize SpellingCorrector with default settings."""
    return SpellingCorrector(case_sensitive=False, count_threshold=5)


class TestSplitWords:
    """Tests for split_words method."""

    def test_split_words_lowercase(self, spelling_corrector: SpellingCorrector):
        """Given a sentence with mixed case, when splitting words, then return lowercase words."""
        # Given
        input_text = "Hello World! This IS a Test."
        expected = ["hello", "world", "this", "is", "a", "test"]

        # When
        result = spelling_corrector.split_words(input_text)

        # Then
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_split_words_with_punctuation(self, spelling_corrector: SpellingCorrector):
        """Given a sentence with punctuation, when splitting words, then ignore punctuation."""
        # Given
        input_text = "Hello, world! How's it going?"
        expected = ["hello", "world", "how", "it", "going"]

        # When
        result = spelling_corrector.split_words(input_text)

        # Then
        assert result == expected, f"Expected {expected}, but got {result}"


class TestUnknownWords:
    """Tests for unknown_words method."""

    def test_unknown_words_with_misspelled(self, spelling_corrector: SpellingCorrector):
        """Given a text with misspelled words, when checking for unknown words, then return misspelled words."""
        # Given
        input_text = "This is a mistke and anothr eror."
        expected = ["mistke", "anothr", "eror"]

        # When
        result = spelling_corrector.unknown_words(input_text)

        # Then
        assert sorted(result) == sorted(
            expected), f"Expected {expected}, but got {result}"

    def test_unknown_words_ignore_numbers(self, spelling_corrector: SpellingCorrector):
        """Given a text with numbers, when checking for unknown words, then exclude numbers."""
        # Given
        input_text = "This is 123 and 456test."
        expected: List[str] = []

        # When
        result = spelling_corrector.unknown_words(input_text)

        # Then
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_unknown_words_with_ignore_words(self, spelling_corrector: SpellingCorrector):
        """Given a text with ignore words, when checking for unknown words, then exclude ignore words."""
        # Given
        spelling_corrector.ignore_words = ["mistke"]
        input_text = "This is a mistke."
        expected = []

        # When
        result = spelling_corrector.unknown_words(input_text)

        # Then
        assert result == expected, f"Expected {expected}, but got {result}"


class TestAutocorrect:
    """Tests for autocorrect method."""

    def test_autocorrect_misspelled_words(self, spelling_corrector: SpellingCorrector):
        """Given a text with misspelled words, when autocorrecting, then return corrected text."""
        # Given
        input_text = "This is a mistke."
        expected = "This is a mistake."

        # When
        result = spelling_corrector.autocorrect(input_text)

        # Then
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_autocorrect_no_changes(self, spelling_corrector: SpellingCorrector):
        """Given a correctly spelled text, when autocorrecting, then return unchanged text."""
        # Given
        input_text = "This is a test."
        expected = "This is a test."

        # When
        result = spelling_corrector.autocorrect(input_text)

        # Then
        assert result == expected, f"Expected {expected}, but got {result}"


class TestSuggestCorrections:
    """Tests for suggest_corrections method."""

    def test_suggest_corrections_for_misspelled(self, spelling_corrector: SpellingCorrector):
        """Given a list of misspelled words, when suggesting corrections, then return correction suggestions."""
        # Given
        misspelled_words = ["mistke", "eror"]
        expected: Dict[str, Optional[Dict[str, int]]] = {
            "mistke": {"mistake": pytest.approx(100, abs=50)},
            "eror": {"error": pytest.approx(100, abs=50)}
        }

        # When
        result = spelling_corrector.suggest_corrections(misspelled_words)

        # Then
        for word in misspelled_words:
            assert word in result, f"Expected {word} in suggestions"
            assert result[word] is not None, f"Expected suggestions for {word}"
            assert list(result[word].keys())[0] == list(expected[word].keys())[0], (
                f"Expected correction {expected[word]} for {word}, but got {result[word]}"
            )

    def test_suggest_corrections_no_candidates(self, spelling_corrector: SpellingCorrector):
        """Given a list with no valid candidates, when suggesting corrections, then return None for suggestions."""
        # Given
        misspelled_words = ["zxcvbnm"]
        expected = {"zxcvbnm": None}

        # When
        result = spelling_corrector.suggest_corrections(misspelled_words)

        # Then
        assert result == expected, f"Expected {expected}, but got {result}"
