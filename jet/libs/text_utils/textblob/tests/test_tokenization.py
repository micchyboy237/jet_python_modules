import pytest
from textblob import TextBlob
from nltk.tokenize import TabTokenizer, BlanklineTokenizer
from typing import List

@pytest.fixture
def textblob():
    """Fixture to provide a TextBlob instance."""
    return TextBlob("Beautiful is better than ugly. Explicit is better than implicit.")

class TestTokenization:
    def test_word_tokenization(self, textblob):
        # Given a TextBlob with multiple sentences
        expected = ['Beautiful', 'is', 'better', 'than', 'ugly', 'Explicit', 'is', 'better', 'than', 'implicit']
        
        # When words are accessed
        result = list(textblob.words)
        
        # Then it returns the correct word list
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_sentence_tokenization(self, textblob):
        # Given a TextBlob with multiple sentences
        expected = ["Beautiful is better than ugly.", "Explicit is better than implicit."]
        
        # When sentences are accessed
        result = [str(s) for s in textblob.sentences]
        
        # Then it returns the correct sentence list
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_custom_tab_tokenizer(self):
        # Given a TextBlob with a custom TabTokenizer
        tokenizer = TabTokenizer()
        textblob = TextBlob("This is\ta rather tabby\tblob.", tokenizer=tokenizer)
        expected = ['This is', 'a rather tabby', 'blob.']
        
        # When tokens are accessed
        result = list(textblob.tokens)
        
        # Then it returns the correct tokens
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_blankline_tokenize_method(self):
        # Given a TextBlob with blankline-separated text
        textblob = TextBlob("A token\n\nof appreciation")
        tokenizer = BlanklineTokenizer()
        expected = ['A token', 'of appreciation']
        
        # When tokenize is called with BlanklineTokenizer
        result = list(textblob.tokenize(tokenizer))
        
        # Then it returns the correct tokens
        assert result == expected, f"Expected {expected}, but got {result}"