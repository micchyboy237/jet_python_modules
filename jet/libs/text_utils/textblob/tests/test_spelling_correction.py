import pytest
from textblob import TextBlob, Word
from typing import List, Tuple

class TestSpellingCorrection:
    def test_textblob_correct(self):
        # Given a TextBlob with misspelled text
        textblob = TextBlob("I havv goood speling!")
        expected = "I have good spelling!"
        
        # When correct is called
        result = str(textblob.correct())
        
        # Then it returns the corrected text
        assert result == expected, f"Expected '{expected}', but got '{result}'"

    def test_word_spellcheck(self):
        # Given a misspelled Word
        word = Word("falibility")
        expected = [('fallibility', 1.0)]
        
        # When spellcheck is called
        result = word.spellcheck()
        
        # Then it returns the correct suggestions
        assert result == expected, f"Expected {expected}, but got {result}"