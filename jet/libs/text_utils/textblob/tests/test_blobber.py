import pytest
from textblob import Blobber
from textblob.taggers import NLTKTagger

@pytest.fixture
def blobber():
    """Fixture to provide a Blobber instance with NLTKTagger."""
    return Blobber(pos_tagger=NLTKTagger())

class TestBlobber:
    def test_blobber_shared_tagger(self, blobber):
        # Given a Blobber instance
        blob1 = blobber("This is a blob.")
        blob2 = blobber("This is another blob.")
        
        # When the pos_tagger attributes are compared
        result = blob1.pos_tagger is blob2.pos_tagger
        
        # Then they are the same instance
        assert result is True, f"Expected pos_taggers to be identical, but they are not"

    def test_blobber_pos_tagging(self, blobber):
        # Given a Blobber instance and a TextBlob
        blob = blobber("Tag! You're It!")
        expected = [('Tag', 'NN'), ('You', 'PRP'), ("'re", 'VBP'), ('It', 'PRP')]
        
        # When pos_tags are accessed
        result = [(word.string, tag) for word, tag in blob.pos_tags]
        
        # Then it returns the correct POS tags
        assert result == expected, f"Expected {expected}, but got {result}"