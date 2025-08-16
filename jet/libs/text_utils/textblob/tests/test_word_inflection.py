import pytest
from textblob import TextBlob, Word
from typing import List

@pytest.fixture
def textblob():
    """Fixture to provide a TextBlob instance."""
    return TextBlob("Use 4 spaces per indentation level.")

class TestWordInflection:
    def test_singularize_word(self, textblob):
        # Given a plural word from a TextBlob
        word = textblob.words[2]
        expected = "space"
        
        # When singularize is called
        result = word.singularize()
        
        # Then it returns the singular form
        assert result == expected, f"Expected '{expected}', but got '{result}'"

    def test_pluralize_word(self, textblob):
        # Given a singular word from a TextBlob
        word = textblob.words[-1]
        expected = "levels"
        
        # When pluralize is called
        result = word.pluralize()
        
        # Then it returns the plural form
        assert result == expected, f"Expected '{expected}', but got '{result}'"

    def test_lemmatize_noun(self):
        # Given a plural noun
        word = Word("octopi")
        expected = "octopus"
        
        # When lemmatize is called
        result = word.lemmatize()
        
        # Then it returns the lemma
        assert result == expected, f"Expected '{expected}', but got '{result}'"

    def test_lemmatize_verb(self):
        # Given a verb in past tense
        word = Word("went")
        expected = "go"
        
        # When lemmatize is called with pos='v'
        result = word.lemmatize("v")
        
        # Then it returns the verb lemma
        assert result == expected, f"Expected '{expected}', but got '{result}'"