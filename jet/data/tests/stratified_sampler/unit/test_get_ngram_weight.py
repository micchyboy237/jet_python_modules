import pytest
from collections import Counter
from jet.data.stratified_sampler import get_ngram_weight

def test_get_ngram_weight_no_previous_ngrams():
    # Given: A Counter of n-grams, sentence n-grams, and no previous n-grams
    all_ngrams = Counter({"Global markets": 2, "markets rise": 1})
    sentence_ngrams = ["Global markets", "markets rise"]
    previous_ngrams = set()
    expected = 1.5
    
    # When: get_ngram_weight is called
    result = get_ngram_weight(all_ngrams, sentence_ngrams, previous_ngrams)
    
    # Then: The result should match the expected weight
    assert abs(result - expected) < 1e-6, f"Expected {expected}, but got {result}"

def test_get_ngram_weight_with_previous_ngrams():
    # Given: A Counter of n-grams, sentence n-grams, and some previous n-grams
    all_ngrams = Counter({"Global markets": 2, "markets rise": 1})
    sentence_ngrams = ["Global markets", "markets rise"]
    previous_ngrams = {"Global markets"}
    expected = 2.5
    
    # When: get_ngram_weight is called
    result = get_ngram_weight(all_ngrams, sentence_ngrams, previous_ngrams)
    
    # Then: The result should match the expected weight
    assert abs(result - expected) < 1e-6, f"Expected {expected}, but got {result}"