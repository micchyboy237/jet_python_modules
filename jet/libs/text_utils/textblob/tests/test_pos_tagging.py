import pytest
from textblob import TextBlob
from textblob.taggers import NLTKTagger
from typing import List, Tuple

@pytest.fixture
def textblob():
    """Fixture to provide a TextBlob instance."""
    return TextBlob("Tag! You're It!")

class TestPOSTagging:
    def test_default_pos_tags(self, textblob):
        # Given a TextBlob with default PatternTagger
        expected = [('Tag', 'NN'), ('You', 'PRP'), ("'re", 'VBP'), ('It', 'PRP')]
        
        # When pos_tags are accessed
        result = [(word.string, tag) for word, tag in textblob.pos_tags]
        
        # Then it returns the correct POS tags
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_nltk_pos_tagger(self):
        # Given a TextBlob with NLTKTagger
        tagger = NLTKTagger()
        textblob = TextBlob("Tag! You're It!", pos_tagger=tagger)
        expected = [('Tag', 'NN'), ('You', 'PRP'), ("'re", 'VBP'), ('It', 'PRP')]
        
        # When pos_tags are accessed
        result = [(word.string, tag) for word, tag in textblob.pos_tags]
        
        # Then it returns the correct POS tags
        assert result == expected, f"Expected {expected}, but got {result}"