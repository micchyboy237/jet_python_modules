import pytest
import inflect
from typing import Optional


@pytest.fixture
def p():
    """Fixture to provide a fresh inflect.engine instance."""
    return inflect.engine()


class TestPluralization:
    def test_plural_noun_singular_to_plural(self, p):
        # Given a singular noun and a count of 2
        word = "person"
        count = 2
        expected = "people"

        # When plural_noun is called
        result = p.plural_noun(word, count)

        # Then it returns the correct plural form
        assert result == expected, f"Expected '{expected}', but got '{result}'"

    def test_plural_noun_singular_with_count_1(self, p):
        # Given a singular noun and a count of 1
        word = "person"
        count = 1
        expected = "person"

        # When plural_noun is called
        result = p.plural_noun(word, count)

        # Then it returns the singular form
        assert result == expected, f"Expected '{expected}', but got '{result}'"

    def test_plural_verb_with_count_1(self, p):
        # Given a singular verb and a count of 1
        word = "was"
        count = 1
        expected = "was"

        # When plural_verb is called
        result = p.plural_verb(word, count)

        # Then it returns the singular verb
        assert result == expected, f"Expected '{expected}', but got '{result}'"

    def test_plural_verb_with_count_2(self, p):
        # Given a singular verb and a count of 2
        word = "was"
        count = 2
        expected = "were"

        # When plural_verb is called
        result = p.plural_verb(word, count)

        # Then it returns the plural verb
        assert result == expected, f"Expected '{expected}', but got '{result}'"

    def test_plural_adj_singular_to_plural(self, p):
        # Given a singular adjective and a count of 2
        word = "my"
        count = 2
        expected = "our"

        # When plural_adj is called
        result = p.plural_adj(word, count)

        # Then it returns the plural adjective
        assert result == expected, f"Expected '{expected}', but got '{result}'"

    def test_singular_noun_plural_to_singular(self, p):
        # Given a plural noun
        word = "people"
        expected = "person"

        # When singular_noun is called
        result = p.singular_noun(word)

        # Then it returns the singular form
        assert result == expected, f"Expected '{expected}', but got '{result}'"

    def test_singular_noun_with_gender_feminine(self, p):
        # Given a plural pronoun and feminine gender
        p.gender("feminine")
        word = "they"
        expected = "she"

        # When singular_noun is called
        result = p.singular_noun(word)

        # Then it returns the feminine singular pronoun
        assert result == expected, f"Expected '{expected}', but got '{result}'"
