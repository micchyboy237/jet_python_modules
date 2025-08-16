import pytest
import inflect


@pytest.fixture
def p():
    """Fixture to provide a fresh inflect.engine instance."""
    return inflect.engine()


@pytest.fixture(autouse=True)
def reset_classical(p):
    """Reset classical settings after each test."""
    yield
    p.classical(all=False)


class TestClassicalInflections:
    def test_classical_plural_noun(self, p):
        # Given classical mode is enabled
        p.classical(all=True)
        word = "focus"
        count = 2
        expected = "foci"

        # When plural_noun is called
        result = p.plural_noun(word, count)

        # Then it returns the classical plural
        assert result == expected, f"Expected '{expected}', but got '{result}'"

    def test_classical_zero_plural(self, p):
        # Given classical zero mode is enabled
        p.classical(zero=True)
        word = "error"
        count = 0
        expected = "no error"

        # When no is called
        result = p.no(word, count)

        # Then it returns the singular form with 'no'
        assert result == expected, f"Expected '{expected}', but got '{result}'"

    def test_classical_herd_plural(self, p):
        # Given classical herd mode is enabled
        p.classical(herd=True)
        word = "buffalo"
        count = 2
        expected = "buffalo"

        # When plural_noun is called
        result = p.plural_noun(word, count)

        # Then it returns the classical plural
        assert result == expected, f"Expected '{expected}', but got '{result}'"

    def test_classical_persons_plural(self, p):
        # Given classical persons mode is enabled
        p.classical(persons=True)
        word = "chairperson"
        count = 2
        expected = "chairpersons"

        # When plural_noun is called
        result = p.plural_noun(word, count)

        # Then it returns the classical plural
        assert result == expected, f"Expected '{expected}', but got '{result}'"
