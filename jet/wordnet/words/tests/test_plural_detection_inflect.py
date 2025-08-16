import pytest
from jet.wordnet.words import is_plural_inflect


@pytest.fixture
def inflect_engine():
    """Fixture to initialize inflect engine."""
    import inflect
    return inflect.engine()


class TestPluralDetectionInflect:
    def test_plural_word(self, inflect_engine):
        # Given a plural word
        word = "cats"
        expected = True

        # When checking if the word is plural
        result = is_plural_inflect(word)

        # Then it should return True
        assert result == expected, f"Expected {word} to be plural, got {result}"

    def test_singular_word(self, inflect_engine):
        # Given a singular word
        word = "dog"
        expected = False

        # When checking if the word is plural
        result = is_plural_inflect(word)

        # Then it should return False
        assert result == expected, f"Expected {word} to be singular, got {result}"

    def test_irregular_plural(self, inflect_engine):
        # Given an irregular plural word
        word = "children"
        expected = True

        # When checking if the word is plural
        result = is_plural_inflect(word)

        # Then it should return True
        assert result == expected, f"Expected {word} to be plural, got {result}"

    def test_uncountable_noun(self, inflect_engine):
        # Given an uncountable noun
        word = "furniture"
        expected = False

        # When checking if the word is plural
        result = is_plural_inflect(word)

        # Then it should return False
        assert result == expected, f"Expected {word} to be singular, got {result}"
