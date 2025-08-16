import pytest
from textblob import TextBlob
from typing import List

@pytest.fixture
def textblob():
    """Fixture to provide a TextBlob instance."""
    return TextBlob("Now is better than never.")

class TestNGrams:
    def test_ngrams_trigrams(self, textblob):
        # Given a TextBlob for trigrams
        expected = [['Now', 'is', 'better'], ['is', 'better', 'than'], ['better', 'than', 'never']]
        
        # When ngrams is called with n=3
        result = [list(ngram) for ngram in textblob.ngrams(n=3)]
        
        # Then it returns the correct trigrams
        assert result == expected, f"Expected {expected}, but got {result}"