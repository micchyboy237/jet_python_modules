import pytest
from unittest.mock import patch
from jet.data.stratified_sampler import StratifiedSampler, ProcessedDataString

def test_filter_strings():
    # Given: A dataset with mixed sentiment categories
    data = [
        ProcessedDataString(source="Global markets rise", category_values=["positive", 1, 5.0, True]),
        ProcessedDataString(source="Global markets stabilize", category_values=["positive", 2, 4.8, True]),
        ProcessedDataString(source="Tech stocks soar", category_values=["negative", 3, 3.5, False])
    ]
    sampler = StratifiedSampler(data, num_samples=2)
    expected = ["Global markets rise", "Global markets stabilize"]
    
    # When: filter_strings is called with n=2 and top_n=2
    with patch("tqdm.tqdm", lambda x, **kwargs: x):
        result = sampler.filter_strings(n=2, top_n=2)
    
    # Then: The result should match the expected filtered strings
    assert result == expected, f"Expected {expected}, but got {result}"