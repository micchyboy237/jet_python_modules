import pytest
from collections import Counter
from jet.data.stratified_sampler import get_ngrams

def test_get_ngrams_single_word():
    # Given: A single word input and n=1
    input_text = "Economy"
    n = 1
    expected = ["Economy"]
    
    # When: get_ngrams is called
    result = get_ngrams(input_text, n)
    
    # Then: The result should match the expected unigram
    assert result == expected, f"Expected {expected}, but got {result}"

def test_get_ngrams_bigrams():
    # Given: A sentence with multiple words and n=2
    input_text = "Global markets rise"
    n = 2
    expected = ["Global markets", "markets rise"]
    
    # When: get_ngrams is called
    result = get_ngrams(input_text, n)
    
    # Then: The result should match the expected bigrams
    assert result == expected, f"Expected {expected}, but got {result}"

def test_get_ngrams_empty_string():
    # Given: An empty string and n=1
    input_text = ""
    n = 1
    expected = []
    
    # When: get_ngrams is called
    result = get_ngrams(input_text, n)
    
    # Then: The result should be an empty list
    assert result == expected, f"Expected {expected}, but got {result}"