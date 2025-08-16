import pytest
import inflect


@pytest.fixture
def p():
    """Fixture to provide a fresh inflect.engine instance."""
    return inflect.engine()


class TestArticles:
    def test_a_with_consonant(self, p):
        # Given a word starting with a consonant
        word = "thing"
        expected = "a thing"

        # When a is called
        result = p.a(word)

        # Then it returns the correct article and word
        assert result == expected, f"Expected '{expected}', but got '{result}'"

    def test_an_with_vowel(self, p):
        # Given a word starting with a vowel
        word = "idea"
        expected = "an idea"

        # When an is called
        result = p.an(word)

        # Then it returns the correct article and word
        assert result == expected, f"Expected '{expected}', but got '{result}'"

    def test_defa_override(self, p):
        # Given a word with a custom article rule
        word = "ape"
        p.defa("ape")
        expected = "a ape"

        # When a is called after overriding
        result = p.a(word)

        # Then it returns the overridden article
        assert result == expected, f"Expected '{expected}', but got '{result}'"

    def test_defan_with_regex(self, p):
        # Given a phrase matching a regex for an
        phrase = "horrendous affectation"
        p.defan("horrendous.*")
        expected = "an horrendous affectation"

        # When a is called
        result = p.a(phrase)

        # Then it returns the correct article based on regex
        assert result == expected, f"Expected '{expected}', but got '{result}'"
