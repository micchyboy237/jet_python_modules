import pytest
import inflect


@pytest.fixture
def p():
    """Fixture to provide a fresh inflect.engine instance."""
    return inflect.engine()


class TestInterpolation:
    def test_inflect_plural_noun(self, p):
        # Given a string with plural interpolation
        template = "The plural of {0} is plural('{0}')"
        word = "car"
        expected = "The plural of car is cars"

        # When inflect is called
        result = p.inflect(template.format(word))

        # Then it returns the correctly interpolated string
        assert result == expected, f"Expected '{expected}', but got '{result}'"

    def test_inflect_with_num_and_plural(self, p):
        # Given a string with num and plural interpolation
        template = "I saw num({0}) plural('cat')"
        count = 3
        expected = "I saw 3 cats"

        # When inflect is called
        result = p.inflect(template.format(count))

        # Then it returns the correctly interpolated string
        assert result == expected, f"Expected '{expected}', but got '{result}'"

    def test_inflect_with_no_and_plural_verb(self, p):
        # Given a string with no and plural_verb interpolation
        template = "There plural_verb('was',{0}) no('error',{0})"
        count = 2
        expected = "There were 2 errors"

        # When inflect is called
        result = p.inflect(template.format(count))

        # Then it returns the correctly interpolated string
        assert result == expected, f"Expected '{expected}', but got '{result}'"
