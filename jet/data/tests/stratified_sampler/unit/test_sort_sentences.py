import pytest
from unittest.mock import patch
from jet.data.stratified_sampler import sort_sentences

def test_sort_sentences_basic():
    # Given: A list of sentences and n=1
    sentences = ["Global markets rise", "Tech stocks soar", "Economy grows steadily"]
    n = 1
    expected = ["Global markets rise", "Tech stocks soar", "Economy grows steadily"]
    
    # When: sort_sentences is called
    with patch("tqdm.tqdm", lambda x, **kwargs: x):
        result = sort_sentences(sentences, n)
    
    # Then: The result should match the expected sentences (order may vary due to set)
    assert set(result) == set(expected), f"Expected {expected}, but got {result}"