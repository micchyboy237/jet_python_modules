import pytest
from collections import Counter
from jet.data.stratified_sampler import calculate_n_gram_diversity

def test_calculate_n_gram_diversity_non_empty():
    # Given: A Counter with n-gram frequencies
    freq = Counter({"Global markets": 1, "markets rise": 2})
    expected = 2
    
    # When: calculate_n_gram_diversity is called
    result = calculate_n_gram_diversity(freq)
    
    # Then: The result should match the expected diversity
    assert result == expected, f"Expected {expected}, but got {result}"

def test_calculate_n_gram_diversity_empty():
    # Given: An empty Counter
    freq = Counter()
    expected = 0
    
    # When: calculate_n_gram_diversity is called
    result = calculate_n_gram_diversity(freq)
    
    # Then: The result should be zero
    assert result == expected, f"Expected {expected}, but got {result}"