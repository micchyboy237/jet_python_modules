import pytest
from unittest.mock import patch
from collections import Counter
from jet.data.stratified_sampler import StratifiedSampler, ProcessedData, ProcessedDataString

def test_get_samples_balance():
    # Given: A dataset with diverse categories and num_samples=4
    data = [
        ProcessedData(source="Global markets rise", target="Positive outlook", category_values=["positive", 1, 5.0, True], score=0.9),
        ProcessedData(source="Tech stocks soar", target="Market boom", category_values=["positive", 2, 4.5, True], score=0.8),
        ProcessedData(source="Economy slows down", target="Negative outlook", category_values=["negative", 3, 3.0, False], score=0.4),
        ProcessedData(source="Markets face uncertainty", target="Cautious approach", category_values=["neutral", 4, 4.0, False], score=0.6),
        ProcessedData(source="Stocks rebound quickly", target="Recovery", category_values=["positive", 5, 4.8, True], score=0.7),
        ProcessedData(source="Financial crisis looms", target="Downturn", category_values=["negative", 6, 2.5, False], score=0.3)
    ]
    sampler = StratifiedSampler(data, num_samples=4)
    expected_len = 4
    expected_sentiments = {"positive": 2, "negative": 1, "neutral": 1}
    expected_ints = {1: 1, 2: 1, 3: 1, 4: 1}
    expected_bools = {True: 2, False: 2}
    
    # When: get_samples is called
    with patch("tqdm.tqdm", lambda x, **kwargs: x):
        result = sampler.get_samples()
    
    # Then: The result should have the expected length and balanced categories
    assert len(result) == expected_len, f"Expected length {expected_len}, but got {len(result)}"
    sentiment_counts = Counter(item['category_values'][0] for item in result)
    assert sentiment_counts == expected_sentiments, f"Expected sentiment balance {expected_sentiments}, but got {sentiment_counts}"
    int_counts = Counter(item['category_values'][1] for item in result)
    assert int_counts == expected_ints or len(int_counts) >= 3, f"Expected integer balance {expected_ints}, but got {int_counts}"
    float_counts = Counter(item['category_values'][2] for item in result)
    assert len(float_counts) >= 3, f"Expected at least 3 float categories, but got {float_counts}"
    bool_counts = Counter(item['category_values'][3] for item in result)
    assert bool_counts == expected_bools, f"Expected boolean balance {expected_bools}, but got {bool_counts}"

def test_split_train_test_val_balance():
    # Given: A dataset with diverse categories and split ratios
    data = [
        ProcessedDataString(source="Global markets rise", category_values=["positive", 1, 5.0, True]),
        ProcessedDataString(source="Tech stocks soar", category_values=["positive", 2, 4.5, True]),
        ProcessedDataString(source="Economy slows down", category_values=["negative", 3, 3.0, False]),
        ProcessedDataString(source="Markets face uncertainty", category_values=["neutral", 4, 4.0, False]),
        ProcessedDataString(source="Stocks rebound quickly", category_values=["positive", 5, 4.8, True]),
        ProcessedDataString(source="Financial crisis looms", category_values=["negative", 6, 2.5, False])
    ]
    sampler = StratifiedSampler(data, num_samples=4)
    expected_len = 6
    expected_sentiments = {'positive': 3, 'negative': 2, 'neutral': 1}
    expected_ints = {1: 1, 2: 1, 3: 1, 4: 1}
    expected_bools = {True: 3, False: 3}
    
    # When: split_train_test_val is called
    with patch("tqdm.tqdm", lambda x, **kwargs: x):
        train_data, test_data, val_data = sampler.split_train_test_val(train_ratio=0.5, test_ratio=0.25)
    
    # Then: The splits should maintain balance and have the expected structure
    all_data = train_data + test_data + val_data
    assert len(all_data) == expected_len, f"Expected total length {expected_len}, but got {len(all_data)}"
    sentiment_counts = Counter(item['category_values'][0] for item in all_data)
    assert sentiment_counts == expected_sentiments, f"Expected sentiment balance {expected_sentiments}, but got {sentiment_counts}"
    int_counts = Counter(item['category_values'][1] for item in all_data)
    assert int_counts == expected_ints or len(int_counts) >= 3, f"Expected integer balance {expected_ints}, but got {int_counts}"
    float_counts = Counter(item['category_values'][2] for item in all_data)
    assert len(float_counts) >= 3, f"Expected at least 3 float categories, but got {float_counts}"
    bool_counts = Counter(item['category_values'][3] for item in all_data)
    assert bool_counts == expected_bools, f"Expected boolean balance {expected_bools}, but got {bool_counts}"