import pytest
from textblob import Word
from textblob.wordnet import VERB, Synset
from typing import List

class TestWordNet:
    def test_word_synsets(self):
        # Given a Word instance
        word = Word("octopus")
        expected = ['octopus.n.01', 'octopus.n.02']
        
        # When synsets are accessed
        result = [s.name() for s in word.synsets]
        
        # Then it returns the correct synset names
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_get_synsets_verb(self):
        # Given a Word with a verb POS
        word = Word("hack")
        expected = ['chop.v.05', 'hack.v.02', 'hack.v.03', 'hack.v.04', 'hack.v.05', 'hack.v.06', 'hack.v.07', 'hack.v.08']
        
        # When get_synsets is called with VERB
        result = [s.name() for s in word.get_synsets(pos=VERB)]
        
        # Then it returns the correct verb synsets
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_word_definitions(self):
        # Given a Word instance
        word = Word("octopus")
        expected = ['tentacles of octopus prepared as food', 'bottom-living cephalopod having a soft oval body with eight long tentacles']
        
        # When definitions are accessed
        result = word.definitions
        
        # Then it returns the correct definitions
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_synset_path_similarity(self):
        # Given two Synset instances
        octopus = Synset("octopus.n.02")
        shrimp = Synset("shrimp.n.03")
        expected = 0.1111111111111111
        
        # When path_similarity is called
        result = octopus.path_similarity(shrimp)
        
        # Then it returns the expected similarity
        assert abs(result - expected) < 1e-10, f"Expected {expected}, but got {result}"