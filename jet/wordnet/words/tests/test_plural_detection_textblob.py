import pytest
from jet.wordnet.words import is_plural_textblob


@pytest.fixture
def textblob_word():
    """Fixture to provide TextBlob Word class."""
    from textblob import Word
    return Word


class TestPluralDetectionTextBlob:
    def test_plural_word(self, textblob_word):
        # Given a plural word
        word = "boxes"
        expected = True

        # When checking if the word is plural
        result = is_plural_textblob(word)

        # Then it should return True
        assert result == expected, f"Expected {word} to be plural, got {result}"

    def test_singular_word(self, textblob_word):
        # Given a singular word
        word = "box"
        expected = False

        # When checking if the word is plural
        result = is_plural_textblob(word)

        # Then it should return False
        assert result == expected, f"Expected {word} to be singular, got {result}"

    def test_irregular_plural(self, textblob_word):
        # Given an irregular plural word
        word = "teeth"
        expected = True

        # When checking if the word is plural
        result = is_plural_textblob(word)

        # Then it should return True
        assert result == expected, f"Expected {word} to be plural, got {result}"

    def test_uncountable_noun(self, textblob_word):
        # Given an uncountable noun
        word = "water"
        expected = False

        # When checking if the word is plural
        result = is_plural_textblob(word)

        # Then it should return False
        assert result == expected, f"Expected {word} to be singular, got {result}"
