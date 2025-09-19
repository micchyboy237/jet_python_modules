import pytest

from jet.utils.text import format_sub_dir

class TestFormatSubDir:
    def test_basic_text_with_spaces(self):
        # Given
        input_text = "Hello World"
        expected = "hello_world"

        # When
        result = format_sub_dir(input_text)

        # Then
        assert result == expected, f"Expected '{expected}', but got '{result}'"

    def test_newlines_replaced_with_underscore(self):
        # Given
        input_text = "Hello\nWorld"
        expected = "hello_world"

        # When
        result = format_sub_dir(input_text)

        # Then
        assert result == expected, f"Expected '{expected}', but got '{result}'"

    def test_multiple_newlines_and_spaces(self):
        # Given
        input_text = "Hello\n\nWorld  Test"
        expected = "hello_world_test"

        # When
        result = format_sub_dir(input_text)

        # Then
        assert result == expected, f"Expected '{expected}', but got '{result}'"

    def test_special_characters(self):
        # Given
        input_text = "Hello@World!#Test"
        expected = "hello_world_test"

        # When
        result = format_sub_dir(input_text)

        # Then
        assert result == expected, f"Expected '{expected}', but got '{result}'"

    def test_consecutive_special_characters_and_newlines(self):
        # Given
        input_text = "Hello!!!\n\nWorld"
        expected = "hello_world"

        # When
        result = format_sub_dir(input_text)

        # Then
        assert result == expected, f"Expected '{expected}', but got '{result}'"

    def test_leading_and_trailing_underscores(self):
        # Given
        input_text = "_Hello_World_"
        expected = "hello_world"

        # When
        result = format_sub_dir(input_text)

        # Then
        assert result == expected, f"Expected '{expected}', but got '{result}'"

    def test_leading_and_trailing_newlines(self):
        # Given
        input_text = "\nHello World\n"
        expected = "hello_world"

        # When
        result = format_sub_dir(input_text)

        # Then
        assert result == expected, f"Expected '{expected}', but got '{result}'"

    def test_only_special_characters(self):
        # Given
        input_text = "!@#$%^"
        expected = ""

        # When
        result = format_sub_dir(input_text)

        # Then
        assert result == expected, f"Expected '{expected}', but got '{result}'"

    def test_empty_string(self):
        # Given
        input_text = ""
        expected = ""

        # When
        result = format_sub_dir(input_text)

        # Then
        assert result == expected, f"Expected '{expected}', but got '{result}'"

    def test_mixed_case_with_numbers(self):
        # Given
        input_text = "Hello123World_TEST"
        expected = "hello123world_test"

        # When
        result = format_sub_dir(input_text)

        # Then
        assert result == expected, f"Expected '{expected}', but got '{result}'"

    def test_consecutive_underscores_from_mixed_sources(self):
        # Given
        input_text = "Hello___World\n\nTest!!"
        expected = "hello_world_test"

        # When
        result = format_sub_dir(input_text)

        # Then
        assert result == expected, f"Expected '{expected}', but got '{result}'"