import pytest
from jet.wordnet.validators.sentence_validator import SentenceValidator
from typing import List


@pytest.fixture
def validator():
    """Fixture to provide a fresh SentenceValidator instance."""
    return SentenceValidator()


class TestSentenceValidator:
    """Test suite for SentenceValidator class."""

    def test_valid_sentence(self, validator: SentenceValidator):
        """Test a valid sentence with noun and verb."""
        # Given
        sentence = "The cat runs quickly."
        expected = True

        # When
        result = validator.is_valid_sentence(sentence)

        # Then
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_missing_verb(self, validator: SentenceValidator):
        """Test a sentence missing a verb."""
        # Given
        sentence = "The cat dog."
        expected = False

        # When
        result = validator.is_valid_sentence(sentence)

        # Then
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_missing_noun(self, validator: SentenceValidator):
        """Test a sentence missing a noun."""
        # Given
        sentence = "Running quickly always."
        expected = False

        # When
        result = validator.is_valid_sentence(sentence)

        # Then
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_too_short_sentence(self, validator: SentenceValidator):
        """Test a sentence that is too short."""
        # Given
        sentence = "Cat runs."
        expected = False

        # When
        result = validator.is_valid_sentence(sentence)

        # Then
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_missing_punctuation(self, validator: SentenceValidator):
        """Test a sentence without proper ending punctuation."""
        # Given
        sentence = "The cat runs quickly"
        expected = False

        # When
        result = validator.is_valid_sentence(sentence)

        # Then
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_empty_sentence(self, validator: SentenceValidator):
        """Test an empty sentence."""
        # Given
        sentence = ""
        expected = False

        # When
        result = validator.is_valid_sentence(sentence)

        # Then
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_invalid_input_type(self, validator: SentenceValidator):
        """Test invalid input type."""
        # Given
        sentence = None
        expected = False

        # When
        result = validator.is_valid_sentence(sentence)

        # Then
        assert result == expected, f"Expected {expected}, but got {result}"
