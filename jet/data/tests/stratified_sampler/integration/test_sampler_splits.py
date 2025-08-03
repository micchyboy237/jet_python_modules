import pytest
from unittest.mock import patch
from jet.data.stratified_sampler import StratifiedSampler, ProcessedData, ProcessedDataString

def test_split_train_test_val_with_processed_data():
    # Given: A dataset with ProcessedData and split ratios
    data = [
        ProcessedData(source="hello", target="world", category_values=["q1"], score=0.9),
        ProcessedData(source="hello again", target="world again", category_values=["q1"], score=0.7),
        ProcessedData(source="test", target="case", category_values=["q2"], score=0.8),
        ProcessedData(source="test again", target="case again", category_values=["q2"], score=0.6),
        ProcessedData(source="sample", target="data", category_values=["q1"], score=0.5)
    ]
    sampler = StratifiedSampler(data, num_samples=3)
    expected_train_len = 3  # 60% of 5
    expected_test_len = 1   # 20% of 5
    expected_val_len = 1    # 20% of 5
    
    # When: split_train_test_val is called
    with patch("tqdm.tqdm", lambda x, **kwargs: x):
        train_data, test_data, val_data = sampler.split_train_test_val(train_ratio=0.6, test_ratio=0.2)
    
    # Then: The splits should have the expected lengths and valid structure
    result_train_len = len(train_data)
    result_test_len = len(test_data)
    result_val_len = len(val_data)
    assert result_train_len == expected_train_len, f"Expected train length {expected_train_len}, got {result_train_len}"
    assert result_test_len == expected_test_len, f"Expected test length {expected_test_len}, got {result_test_len}"
    assert result_val_len == expected_val_len, f"Expected val length {expected_val_len}, got {result_val_len}"
    assert all(isinstance(item, dict) for item in train_data + test_data + val_data), "Result contains non-dict items"
    assert all('target' in item for item in train_data + test_data + val_data), "Missing target in ProcessedData"

def test_split_train_test_val_with_processed_data_string():
    # Given: A dataset with ProcessedDataString and split ratios
    data = [
        ProcessedDataString(source="https://example.com/page1", category_values=["example.com"]),
        ProcessedDataString(source="https://example.com/page2", category_values=["example.com"]),
        ProcessedDataString(source="https://test.org/path", category_values=["test.org"]),
        ProcessedDataString(source="https://test.org/other", category_values=["test.org"]),
        ProcessedDataString(source="https://blog.io/post", category_values=["blog.io"])
    ]
    sampler = StratifiedSampler(data, num_samples=3)
    expected_train_len = 3  # 60% of 5
    expected_test_len = 1   # 20% of 5
    expected_val_len = 1    # 20% of 5
    
    # When: split_train_test_val is called
    with patch("tqdm.tqdm", lambda x, **kwargs: x):
        train_data, test_data, val_data = sampler.split_train_test_val(train_ratio=0.6, test_ratio=0.2)
    
    # Then: The splits should have the expected lengths and valid structure
    result_train_len = len(train_data)
    result_test_len = len(test_data)
    result_val_len = len(val_data)
    assert result_train_len == expected_train_len, f"Expected train length {expected_train_len}, got {result_train_len}"
    assert result_test_len == expected_test_len, f"Expected test length {expected_test_len}, got {result_test_len}"
    assert result_val_len == expected_val_len, f"Expected val length {expected_val_len}, got {result_val_len}"
    assert all(isinstance(item, dict) for item in train_data + test_data + val_data), "Result contains non-dict items"
    assert all('source' in item and 'category_values' in item for item in train_data + test_data + val_data), "Missing required fields"
    assert all('target' not in item for item in train_data + test_data + val_data), "Unexpected target in ProcessedDataString"