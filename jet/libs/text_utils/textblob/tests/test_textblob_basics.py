import pytest
from textblob import TextBlob
from typing import List, Tuple

@pytest.fixture
def textblob():
    """Fixture to provide a TextBlob instance."""
    return TextBlob("Python is a high-level programming language.")

class TestTextBlobBasics:
    def test_create_textblob(self):
        # Given a text string
        text = "Python is a high-level programming language."
        expected = text
        
        # When a TextBlob is created
        result = TextBlob(text).string
        
        # Then it matches the input text
        assert result == expected, f"Expected '{expected}', but got '{result}'"

    def test_string_operations_upper(self, textblob):
        # Given a TextBlob instance
        expected = "PYTHON IS A HIGH-LEVEL PROGRAMMING LANGUAGE."
        
        # When the upper method is called
        result = textblob.upper().string
        
        # Then it returns the uppercase text
        assert result == expected, f"Expected '{expected}', but got '{result}'"

    def test_string_comparison(self):
        # Given two TextBlobs
        apple_blob = TextBlob("apples")
        banana_blob = TextBlob("bananas")
        expected = True
        
        # When comparing them lexicographically
        result = apple_blob < banana_blob
        
        # Then it returns the correct comparison
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_concatenation(self):
        # Given two TextBlobs
        apple_blob = TextBlob("apples")
        banana_blob = TextBlob("bananas")
        expected = "apples and bananas"
        
        # When concatenating with a string
        result = (apple_blob + " and " + banana_blob).string
        
        # Then it returns the concatenated string
        assert result == expected, f"Expected '{expected}', but got '{result}'"