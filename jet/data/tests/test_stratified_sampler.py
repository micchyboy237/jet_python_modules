import pytest
import numpy as np
from collections import Counter, defaultdict
from typing import List, Union
from unittest.mock import patch
from jet.data.stratified_sampler import (
    StratifiedSampler, ProcessedData, ProcessedDataString, StratifiedData,
    get_ngrams, get_ngram_weight, sort_sentences, n_gram_frequency,
    calculate_n_gram_diversity, filter_and_sort_sentences_by_ngrams
)


class TestGetNgrams:
    def test_get_ngrams_single_word(self):
        input_text = "Economy"
        n = 1
        expected = ["Economy"]
        result = get_ngrams(input_text, n)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_get_ngrams_bigrams(self):
        input_text = "Global markets rise"
        n = 2
        expected = ["Global markets", "markets rise"]
        result = get_ngrams(input_text, n)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_get_ngrams_empty_string(self):
        input_text = ""
        n = 1
        expected = []
        result = get_ngrams(input_text, n)
        assert result == expected, f"Expected {expected}, but got {result}"


class TestGetNgramWeight:
    def test_get_ngram_weight_no_previous_ngrams(self):
        all_ngrams = Counter({"Global markets": 2, "markets rise": 1})
        sentence_ngrams = ["Global markets", "markets rise"]
        previous_ngrams = set()
        expected = 1.5
        result = get_ngram_weight(all_ngrams, sentence_ngrams, previous_ngrams)
        assert abs(
            result - expected) < 1e-6, f"Expected {expected}, but got {result}"

    def test_get_ngram_weight_with_previous_ngrams(self):
        all_ngrams = Counter({"Global markets": 2, "markets rise": 1})
        sentence_ngrams = ["Global markets", "markets rise"]
        previous_ngrams = {"Global markets"}
        expected = 2.5
        result = get_ngram_weight(all_ngrams, sentence_ngrams, previous_ngrams)
        assert abs(
            result - expected) < 1e-6, f"Expected {expected}, but got {result}"


class TestSortSentences:
    def test_sort_sentences_basic(self):
        sentences = ["Global markets rise",
                     "Tech stocks soar", "Economy grows steadily"]
        n = 1
        expected = ["Global markets rise",
                    "Tech stocks soar", "Economy grows steadily"]
        with patch("tqdm.tqdm", lambda x, **kwargs: x):
            result = sort_sentences(sentences, n)
        assert set(result) == set(
            expected), f"Expected {expected}, but got {result}"


class TestNGramFrequency:
    def test_n_gram_frequency_bigrams(self):
        sentence = "Global markets Global"
        n = 2
        expected = Counter({
            'Gl': 2, 'lo': 2, 'ob': 2, 'ba': 2, 'al': 2, 'l ': 1,
            ' m': 1, 'ma': 1, 'ar': 1, 'rk': 1, 'ke': 1, 'et': 1,
            'ts': 1, 's ': 1, ' G': 1
        })
        result = n_gram_frequency(sentence, n)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_n_gram_frequency_single_char(self):
        sentence = "abc"
        n = 2
        expected = Counter({"ab": 1, "bc": 1})
        result = n_gram_frequency(sentence, n)
        assert result == expected, f"Expected {expected}, but got {result}"


class TestCalculateNGramDiversity:
    def test_calculate_n_gram_diversity(self):
        freq = Counter({"Global markets": 1, "markets rise": 2})
        expected = 2
        result = calculate_n_gram_diversity(freq)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_calculate_n_gram_diversity_empty(self):
        freq = Counter()
        expected = 0
        result = calculate_n_gram_diversity(freq)
        assert result == expected, f"Expected {expected}, but got {result}"


class TestFilterAndSortSentencesByNgrams:
    def test_filter_and_sort_sentences_by_ngrams(self):
        sentences = ["Global markets rise",
                     "Global markets stabilize", "Tech stocks soar"]
        n = 2
        top_n = 2
        is_start_ngrams = True
        with patch("tqdm.tqdm", lambda x, **kwargs: x):
            result = filter_and_sort_sentences_by_ngrams(
                sentences, n, top_n, is_start_ngrams)
        expected = ["Global markets rise", "Global markets stabilize"]
        assert result == expected, f"Expected {expected}, but got {result}"


class TestStratifiedSampler:
    def test_init_with_float_num_samples(self):
        data = [
            ProcessedDataString(source="Global markets rise", category_values=[
                                "positive", 1, 5.0, True]),
            ProcessedDataString(source="Tech stocks soar",
                                category_values=["positive", 2, 4.5, True])
        ]
        num_samples = 0.5
        sampler = StratifiedSampler(data, num_samples)
        expected = 1
        result = sampler.num_samples
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_init_with_invalid_num_samples(self):
        data = [ProcessedDataString(
            source="Global markets rise", category_values=["positive", 1, 5.0, True])]
        num_samples = 0.0
        expected = ValueError
        with pytest.raises(expected):
            StratifiedSampler(data, num_samples)

    def test_init_with_invalid_category_values(self):
        data = [ProcessedDataString(
            source="Global markets rise", category_values=["positive", 1, [5.0], True])]
        num_samples = 1
        expected = ValueError
        with pytest.raises(expected):
            StratifiedSampler(data, num_samples)

    def test_filter_strings(self):
        data = [
            ProcessedDataString(source="Global markets rise", category_values=[
                                "positive", 1, 5.0, True]),
            ProcessedDataString(source="Global markets stabilize", category_values=[
                                "positive", 2, 4.8, True]),
            ProcessedDataString(source="Tech stocks soar", category_values=[
                                "negative", 3, 3.5, False])
        ]
        sampler = StratifiedSampler(data, num_samples=2)
        with patch("tqdm.tqdm", lambda x, **kwargs: x):
            result = sampler.filter_strings(n=2, top_n=2)
        expected = ["Global markets rise", "Global markets stabilize"]
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_get_samples(self):
        data = [
            ProcessedData(source="Global markets rise", target="Positive outlook",
                          category_values=["positive", 1, 5.0, True], score=0.9),
            ProcessedData(source="Tech stocks soar", target="Market boom", category_values=[
                          "positive", 2, 4.5, True], score=0.8),
            ProcessedData(source="Economy slows down", target="Negative outlook",
                          category_values=["negative", 3, 3.0, False], score=0.4),
            ProcessedData(source="Markets face uncertainty", target="Cautious approach",
                          category_values=["neutral", 4, 4.0, False], score=0.6)
        ]
        sampler = StratifiedSampler(data, num_samples=2)
        with patch("tqdm.tqdm", lambda x, **kwargs: x):
            result = sampler.get_samples()
        expected_sources = ["Global markets rise", "Tech stocks soar",
                            "Economy slows down", "Markets face uncertainty"]
        expected_len = 2
        assert len(
            result) == expected_len, f"Expected length {expected_len}, but got {len(result)}"
        assert all(isinstance(item, dict)
                   for item in result), "Result contains non-dict items"
        assert all(
            item["source"] in expected_sources for item in result), f"Invalid source in result: {result}"

    def test_get_unique_strings(self):
        data = [
            ProcessedDataString(source="Global markets rise", category_values=[
                                "positive", 1, 5.0, True]),
            ProcessedDataString(source="Tech stocks soar", category_values=[
                                "negative", 2, 4.0, False])
        ]
        sampler = StratifiedSampler(data, num_samples=1)
        with patch("tqdm.tqdm", lambda x, **kwargs: x):
            result = sampler.get_unique_strings()
        expected = ["Global markets rise", "Tech stocks soar"]
        assert len(result) == 1, f"Expected length 1, but got {len(result)}"
        assert result[0] in expected, f"Expected one of {expected}, but got {result}"

    def test_load_data_with_labels(self):
        data = ["Global markets rise", "Tech stocks soar"]
        sampler = StratifiedSampler(data, num_samples=1)
        with patch("tqdm.tqdm", lambda x, **kwargs: x):
            result = sampler.load_data_with_labels(max_q=2)
        expected = 2
        assert len(
            result) == expected, f"Expected length {expected}, but got {len(result)}"
        assert all(isinstance(item, dict)
                   for item in result), "Result contains non-dict items"
        assert all(len(item['category_values']) ==
                   4 for item in result), "Category values length incorrect"

    def test_split_train_test_val_with_processed_data(self):
        data = [
            ProcessedData(source="hello", target="world",
                          category_values=["q1"], score=0.9),
            ProcessedData(source="hello again", target="world again",
                          category_values=["q1"], score=0.7),
            ProcessedData(source="test", target="case",
                          category_values=["q2"], score=0.8),
            ProcessedData(source="test again", target="case again",
                          category_values=["q2"], score=0.6),
            ProcessedData(source="sample", target="data",
                          category_values=["q1"], score=0.5)
        ]
        sampler = StratifiedSampler(data, num_samples=3)
        with patch("tqdm.tqdm", lambda x, **kwargs: x):
            train_data, test_data, val_data = sampler.split_train_test_val(
                train_ratio=0.6, test_ratio=0.2)
        expected_train_len = 3  # 60% of 5
        expected_test_len = 1   # 20% of 5
        expected_val_len = 1    # 20% of 5
        result_train_len = len(train_data)
        result_test_len = len(test_data)
        result_val_len = len(val_data)
        assert result_train_len == expected_train_len, f"Expected train length {expected_train_len}, got {result_train_len}"
        assert result_test_len == expected_test_len, f"Expected test length {expected_test_len}, got {result_test_len}"
        assert result_val_len == expected_val_len, f"Expected val length {expected_val_len}, got {result_val_len}"
        assert all(isinstance(item, dict) for item in train_data +
                   test_data + val_data), "Result contains non-dict items"
        assert all('target' in item for item in train_data +
                   test_data + val_data), "Missing target in ProcessedData"

    def test_split_train_test_val_with_processed_data_string(self):
        data = [
            ProcessedDataString(
                source="https://example.com/page1", category_values=["example.com"]),
            ProcessedDataString(
                source="https://example.com/page2", category_values=["example.com"]),
            ProcessedDataString(source="https://test.org/path",
                                category_values=["test.org"]),
            ProcessedDataString(source="https://test.org/other",
                                category_values=["test.org"]),
            ProcessedDataString(source="https://blog.io/post",
                                category_values=["blog.io"])
        ]
        sampler = StratifiedSampler(data, num_samples=3)
        with patch("tqdm.tqdm", lambda x, **kwargs: x):
            train_data, test_data, val_data = sampler.split_train_test_val(
                train_ratio=0.6, test_ratio=0.2)
        expected_train_len = 3  # 60% of 5
        expected_test_len = 1   # 20% of 5
        expected_val_len = 1    # 20% of 5
        result_train_len = len(train_data)
        result_test_len = len(test_data)
        result_val_len = len(val_data)
        assert result_train_len == expected_train_len, f"Expected train length {expected_train_len}, got {result_train_len}"
        assert result_test_len == expected_test_len, f"Expected test length {expected_test_len}, got {result_test_len}"
        assert result_val_len == expected_val_len, f"Expected val length {expected_val_len}, got {result_val_len}"
        assert all(isinstance(item, dict) for item in train_data +
                   test_data + val_data), "Result contains non-dict items"
        assert all('source' in item and 'category_values' in item for item in train_data +
                   test_data + val_data), "Missing required fields"
        assert all('target' not in item for item in train_data +
                   test_data + val_data), "Unexpected target in ProcessedDataString"


class TestStratifiedSamplerBalance:
    def test_get_samples_balance(self):
        data = [
            ProcessedData(source="Global markets rise", target="Positive outlook",
                          category_values=["positive", 1, 5.0, True], score=0.9),
            ProcessedData(source="Tech stocks soar", target="Market boom", category_values=[
                          "positive", 2, 4.5, True], score=0.8),
            ProcessedData(source="Economy slows down", target="Negative outlook",
                          category_values=["negative", 3, 3.0, False], score=0.4),
            ProcessedData(source="Markets face uncertainty", target="Cautious approach",
                          category_values=["neutral", 4, 4.0, False], score=0.6),
            ProcessedData(source="Stocks rebound quickly", target="Recovery", category_values=[
                          "positive", 5, 4.8, True], score=0.7),
            ProcessedData(source="Financial crisis looms", target="Downturn", category_values=[
                          "negative", 6, 2.5, False], score=0.3)
        ]
        sampler = StratifiedSampler(data, num_samples=4)
        with patch("tqdm.tqdm", lambda x, **kwargs: x):
            result = sampler.get_samples()
        expected_len = 4
        assert len(
            result) == expected_len, f"Expected length {expected_len}, but got {len(result)}"
        # Check balance of sentiment categories (index 0)
        sentiment_counts = Counter(
            item['category_values'][0] for item in result)
        expected_sentiments = {"positive": 2, "negative": 1, "neutral": 1}
        assert sentiment_counts == expected_sentiments, f"Expected sentiment balance {expected_sentiments}, but got {sentiment_counts}"
        # Check balance of integer categories (index 1)
        int_counts = Counter(item['category_values'][1] for item in result)
        expected_ints = {1: 1, 2: 1, 3: 1, 4: 1}
        assert int_counts == expected_ints or len(
            int_counts) >= 3, f"Expected integer balance {expected_ints}, but got {int_counts}"
        # Check balance of float categories (index 2)
        float_counts = Counter(item['category_values'][2] for item in result)
        assert len(
            float_counts) >= 3, f"Expected at least 3 float categories, but got {float_counts}"
        # Check balance of boolean categories (index 3)
        bool_counts = Counter(item['category_values'][3] for item in result)
        expected_bools = {True: 2, False: 2}
        assert bool_counts == expected_bools, f"Expected boolean balance {expected_bools}, but got {bool_counts}"

    def test_split_train_test_val_balance(self):
        data = [
            ProcessedDataString(source="Global markets rise", category_values=[
                                "positive", 1, 5.0, True]),
            ProcessedDataString(source="Tech stocks soar", category_values=[
                                "positive", 2, 4.5, True]),
            ProcessedDataString(source="Economy slows down", category_values=[
                                "negative", 3, 3.0, False]),
            ProcessedDataString(source="Markets face uncertainty", category_values=[
                                "neutral", 4, 4.0, False]),
            ProcessedDataString(source="Stocks rebound quickly", category_values=[
                                "positive", 5, 4.8, True]),
            ProcessedDataString(source="Financial crisis looms", category_values=[
                                "negative", 6, 2.5, False])
        ]
        sampler = StratifiedSampler(data, num_samples=4)
        with patch("tqdm.tqdm", lambda x, **kwargs: x):
            train_data, test_data, val_data = sampler.split_train_test_val(
                train_ratio=0.5, test_ratio=0.25)
        all_data = train_data + test_data + val_data
        expected_len = 6
        assert len(
            all_data) == expected_len, f"Expected total length {expected_len}, but got {len(all_data)}"
        # Check balance of sentiment categories (index 0)
        sentiment_counts = Counter(
            item['category_values'][0] for item in all_data)
        expected_sentiments = {'positive': 3, 'negative': 2, 'neutral': 1}
        assert sentiment_counts == expected_sentiments, f"Expected sentiment balance {expected_sentiments}, but got {sentiment_counts}"
        # Check balance of integer categories (index 1)
        int_counts = Counter(item['category_values'][1] for item in all_data)
        expected_ints = {1: 1, 2: 1, 3: 1, 4: 1}
        assert int_counts == expected_ints or len(
            int_counts) >= 3, f"Expected integer balance {expected_ints}, but got {int_counts}"
        # Check balance of float categories (index 2)
        float_counts = Counter(item['category_values'][2] for item in all_data)
        assert len(
            float_counts) >= 3, f"Expected at least 3 float categories, but got {float_counts}"
        # Check balance of boolean categories (index 3)
        bool_counts = Counter(item['category_values'][3] for item in all_data)
        expected_bools = {True: 3, False: 3}
        assert bool_counts == expected_bools, f"Expected boolean balance {expected_bools}, but got {bool_counts}"
