import pytest
from collections import Counter
from jet.data.stratified_sampler import n_gram_frequency

def test_n_gram_frequency_bigrams():
    # Given: A sentence with repeated words and n=2
    sentence = "Global markets Global"
    n = 2
    expected = Counter({
        'Gl': 2, 'lo': 2, 'ob': 2, 'ba': 2, 'al': 2, 'l ': 1,
        ' m': 1, 'ma': 1, 'ar': 1, 'rk': 1, 'ke': 1, 'et': 1,
        'ts': 1, 's ': 1, ' G': 1
    })
    
    # When: n_gram_frequency is called
    result = n_gram_frequency(sentence, n)
    
    # Then: The result should match the expected frequency counter
    assert result == expected, f"Expected {expected}, but got {result}"

def test_n_gram_frequency_single_char():
    # Given: A short string and n=2
    sentence = "abc"
    n = 2
    expected = Counter({"ab": 1, "bc": 1})
    
    # When: n_gram_frequency is called
    result = n_gram_frequency(sentence, n)
    
    # Then: The result should match the expected frequency counter
    assert result == expected, f"Expected {expected}, but got {result}"