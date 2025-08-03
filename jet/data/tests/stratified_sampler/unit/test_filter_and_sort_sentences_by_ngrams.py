import pytest
from unittest.mock import patch
from jet.data.stratified_sampler import filter_and_sort_sentences_by_ngrams

def test_filter_and_sort_sentences_by_ngrams():
    # Given: A list of sentences, n=2, top_n=2, and start n-grams
    sentences = ["Global markets rise", "Global markets stabilize", "Tech stocks soar"]
    n = 2
    top_n = 2
    is_start_ngrams = True
    expected = ["Global markets rise", "Global markets stabilize"]
    
    # When: filter_and_sort_sentences_by_ngrams is called
    with patch("tqdm.tqdm", lambda x, **kwargs: x):
        result = filter_and_sort_sentences_by_ngrams(sentences, n, top_n, is_start_ngrams)
    
    # Then: The result should match the expected filtered and sorted sentences
    assert result == expected, f"Expected {expected}, but got {result}"