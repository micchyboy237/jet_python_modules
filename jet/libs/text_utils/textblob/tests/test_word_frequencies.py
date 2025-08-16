import pytest
from textblob import TextBlob
from typing import int

@pytest.fixture
def textblob():
    """Fixture to provide a TextBlob instance."""
    return TextBlob("We are no longer the Knights who say Ni. We are now the Knights who say Ekki ekki ekki PTANG.")

class TestWordFrequencies:
    def test_word_counts(self, textblob):
        # Given a TextBlob with repeated words
        word = "ekki"
        expected = 3
        
        # When word_counts is accessed
        result = textblob.word_counts[word]
        
        # Then it returns the correct frequency
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_count_case_insensitive(self, textblob):
        # Given a TextBlob with mixed-case words
        word = "ekki"
        expected = 3
        
        # When count is called
        result = textblob.words.count(word)
        
        # Then it returns the correct frequency
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_count_case_sensitive(self, textblob):
        # Given a TextBlob with mixed-case words
        word = "ekki"
        expected = 2
        
        # When count is called with case_sensitive=True
        result = textblob.words.count(word, case_sensitive=True)
        
        # Then it returns the correct frequency
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_noun_phrase_count(self, textblob):
        # Given a TextBlob with noun phrases
        textblob = TextBlob("Python is a python programming language.")
        phrase = "python"
        expected = 2
        
        # When noun_phrases.count is called
        result = textblob.noun_phrases.count(phrase)
        
        # Then it returns the correct frequency
        assert result == expected, f"Expected {expected}, but got {result}"