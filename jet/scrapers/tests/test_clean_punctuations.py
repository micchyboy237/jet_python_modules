import pytest
import re
from typing import List, Tuple
from jet.scrapers.utils import clean_punctuations


class TestCleanPunctuations:
    @pytest.mark.parametrize(
        "input_text, expected",
        [
            (
                # Given: A string with consecutive punctuation marks
                "Hello!!! How are you???",
                # Then: Should keep only the last punctuation mark
                "Hello! How are you?"
            ),
            (
                # Given: A string with mixed consecutive punctuation
                "Wait...?! What.!?",
                # Then: Should keep the last punctuation mark in each group
                "Wait! What?"
            ),
            (
                # Given: A string with punctuation between words
                "anime-strongest",
                # Then: Should replace punctuation with a space
                "anime strongest"
            ),
            (
                # Given: A string with multiple types of punctuation between words
                "data.test,123",
                # Then: Should replace all punctuation with spaces
                "data test 123"
            ),
            (
                # Given: A string with punctuation and numbers
                "summer,2024",
                # Then: Should replace punctuation with a space
                "summer 2024"
            ),
            (
                # Given: A string with mixed punctuation and consecutive marks
                "Really...?!? Are-you.sure??",
                # Then: Should handle both consecutive punctuation and punctuation between words
                "Really? Are you sure?"
            ),
            (
                # Given: A string with no punctuation
                "Hello world",
                # Then: Should return unchanged
                "Hello world"
            ),
            (
                # Given: An empty string
                "",
                # Then: Should return empty string
                ""
            )
        ]
    )
    def test_clean_punctuations(self, input_text: str, expected: str) -> None:
        # Given: An input string with various punctuation patterns
        # When: The clean_punctuations function is called
        result = clean_punctuations(input_text)
        # Then: The result should match the expected output
        assert result == expected, f"Expected '{expected}', but got '{result}'"
