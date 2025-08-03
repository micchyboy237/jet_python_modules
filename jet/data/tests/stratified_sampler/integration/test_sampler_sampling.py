import pytest
from unittest.mock import patch
from collections import Counter
from jet.data.stratified_sampler import StratifiedSampler, ProcessedData, ProcessedDataString

def test_get_samples():
    # Given: A dataset with mixed categories and num_samples=2
    data = [
        ProcessedData(source="Global markets rise", target="Positive outlook", category_values=["positive", 1, 5.0, True], score=0.9),
        ProcessedData(source="Tech stocks soar", target="Market boom", category_values=["positive", 2, 4.5, True], score=0.8),
        ProcessedData(source="Economy slows down", target="Negative outlook", category_values=["negative", 3, 3.0, False], score=0.4),
        ProcessedData(source="Markets face uncertainty", target="Cautious approach", category_values=["neutral", 4, 4.0, False], score=0.6)
    ]
    sampler = StratifiedSampler(data, num_samples=2)
    expected_sources = ["Global markets rise", "Tech stocks soar", "Economy slows down", "Markets face uncertainty"]
    expected_len = 2
    
    # When: get_samples is called
    with patch("tqdm.tqdm", lambda x, **kwargs: x):
        result = sampler.get_samples()
    
    # Then: The result should have the expected length and valid sources
    assert len(result) == expected_len, f"Expected length {expected_len}, but got {len(result)}"
    assert all(isinstance(item, dict) for item in result), "Result contains non-dict items"
    assert all(item["source"] in expected_sources for item in result), f"Invalid source in result: {result}"

def test_get_unique_strings():
    # Given: A dataset with two strings and num_samples=1
    data = [
        ProcessedDataString(source="Global markets rise", category_values=["positive", 1, 5.0, True]),
        ProcessedDataString(source="Tech stocks soar", category_values=["negative", 2, 4.0, False])
    ]
    sampler = StratifiedSampler(data, num_samples=1)
    expected = ["Global markets rise", "Tech stocks soar"]
    
    # When: get_unique_strings is called
    with patch("tqdm.tqdm", lambda x, **kwargs: x):
        result = sampler.get_unique_strings()
    
    # Then: The result should have length 1 and contain a valid string
    assert len(result) == 1, f"Expected length 1, but got {len(result)}"
    assert result[0] in expected, f"Expected one of {expected}, but got {result}"

def test_load_data_with_labels():
    # Given: A list of strings and max_q=2
    data = ["Global markets rise", "Tech stocks soar"]
    sampler = StratifiedSampler(data, num_samples=1)
    expected_len = 2
    
    # When: load_data_with_labels is called
    with patch("tqdm.tqdm", lambda x, **kwargs: x):
        result = sampler.load_data_with_labels(max_q=2)
    
    # Then: The result should have the expected length and valid structure
    assert len(result) == expected_len, f"Expected length {expected_len}, but got {len(result)}"
    assert all(isinstance(item, dict) for item in result), "Result contains non-dict items"
    assert all(len(item['category_values']) == 4 for item in result), "Category values length incorrect"