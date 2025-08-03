import pytest
from jet.data.stratified_sampler import StratifiedSampler, ProcessedDataString

def test_init_with_float_num_samples():
    # Given: A dataset with ProcessedDataString and a float num_samples
    data = [
        ProcessedDataString(source="Global markets rise", category_values=["positive", 1, 5.0, True]),
        ProcessedDataString(source="Tech stocks soar", category_values=["positive", 2, 4.5, True])
    ]
    num_samples = 0.5
    expected = 1
    
    # When: StratifiedSampler is initialized
    sampler = StratifiedSampler(data, num_samples)
    
    # Then: The num_samples should be rounded to the expected integer
    result = sampler.num_samples
    assert result == expected, f"Expected {expected}, but got {result}"

def test_init_with_invalid_num_samples():
    # Given: A dataset and an invalid (zero) num_samples
    data = [ProcessedDataString(source="Global markets rise", category_values=["positive", 1, 5.0, True])]
    num_samples = 0.0
    expected = ValueError
    
    # When: StratifiedSampler is initialized
    # Then: A ValueError should be raised
    with pytest.raises(expected):
        StratifiedSampler(data, num_samples)

def test_init_with_invalid_category_values():
    # Given: A dataset with invalid category_values (contains a list)
    data = [ProcessedDataString(source="Global markets rise", category_values=["positive", 1, [5.0], True])]
    num_samples = 1
    expected = ValueError
    
    # When: StratifiedSampler is initialized
    # Then: A ValueError should be raised
    with pytest.raises(expected):
        StratifiedSampler(data, num_samples)