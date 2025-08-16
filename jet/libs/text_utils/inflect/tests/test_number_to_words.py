import pytest
import inflect
from typing import Union, List


@pytest.fixture
def p():
    """Fixture to provide a fresh inflect.engine instance."""
    return inflect.engine()


class TestNumberToWords:
    def test_number_to_words_single_digit(self, p):
        # Given a single-digit number
        number = 1
        expected = "one"

        # When number_to_words is called
        result = p.number_to_words(number)

        # Then it returns the word representation
        assert result == expected, f"Expected '{expected}', but got '{result}'"

    def test_number_to_words_multi_digit(self, p):
        # Given a multi-digit number
        number = 1234
        expected = "one thousand, two hundred and thirty-four"

        # When number_to_words is called
        result = p.number_to_words(number)

        # Then it returns the correct word representation
        assert result == expected, f"Expected '{expected}', but got '{result}'"

    def test_number_to_words_with_wantlist(self, p):
        # Given a number with wantlist=True
        number = 1234
        expected = ["one thousand", "two hundred and thirty-four"]

        # When number_to_words is called with wantlist
        result = p.number_to_words(number, wantlist=True)

        # Then it returns a list of word parts
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_number_to_words_with_group_1(self, p):
        # Given a number with group=1
        number = 12345
        expected = "one, two, three, four, five"

        # When number_to_words is called with group=1
        result = p.number_to_words(number, group=1)

        # Then it returns digits as individual words
        assert result == expected, f"Expected '{expected}', but got '{result}'"

    def test_number_to_words_with_threshold(self, p):
        # Given a number above the threshold
        number = 11
        threshold = 10
        expected = "11"

        # When number_to_words is called with threshold
        result = p.number_to_words(number, threshold=threshold)

        # Then it returns the numeral as a string
        assert result == expected, f"Expected '{expected}', but got '{result}'"

    def test_ordinal_conversion(self, p):
        # Given a number
        number = 3
        expected = "3rd"

        # When ordinal is called
        result = p.ordinal(number)

        # Then it returns the correct ordinal form
        assert result == expected, f"Expected '{expected}', but got '{result}'"
