import pytest
from textblob import TextBlob
from textblob.np_extractors import ConllExtractor
from typing import List

@pytest.fixture
def textblob():
    """Fixture to provide a TextBlob instance."""
    return TextBlob("Python is a high-level programming language.")

class TestNounPhrases:
    def test_default_noun_phrases(self, textblob):
        # Given a TextBlob with default FastNPExtractor
        expected = ['python']
        
        # When noun_phrases are accessed
        result = list(textblob.noun_phrases)
        
        # Then it returns the correct noun phrases
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_conll_noun_phrases(self):
        # Given a TextBlob with ConllExtractor
        extractor = ConllExtractor()
        textblob = TextBlob("Python is a high-level programming language.", np_extractor=extractor)
        expected = ['python', 'high-level programming language']
        
        # When noun_phrases are accessed
        result = list(textblob.noun_phrases)
        
        # Then it returns the correct noun phrases
        assert result == expected, f"Expected {expected}, but got {result}"