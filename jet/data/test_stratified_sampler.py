import pytest
import numpy as np
from collections import Counter, defaultdict
from typing import List
from unittest.mock import patch
from .stratified_sampler import (
    StratifiedSampler, ProcessedData, ProcessedDataString, StratifiedData,
    get_ngrams, get_ngram_weight, sort_sentences, n_gram_frequency,
    calculate_n_gram_diversity, filter_and_sort_sentences_by_ngrams
)


class TestGetNgrams:
    def test_get_ngrams_single_word(self):
        input_text = "hello"
        n = 1
        expected = ["hello"]
        result = get_ngrams(input_text, n)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_get_ngrams_bigrams(self):
        input_text = "hello world test"
        n = 2
        expected = ["hello world", "world test"]
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
        all_ngrams = Counter({"hello world": 2, "world test": 1})
        sentence_ngrams = ["hello world", "world test"]
        previous_ngrams = set()
        expected = 1.5  # 1/2 + 1/1 = 0.5 + 1.0
        result = get_ngram_weight(all_ngrams, sentence_ngrams, previous_ngrams)
        assert abs(
            result - expected) < 1e-6, f"Expected {expected}, but got {result}"

    def test_get_ngram_weight_with_previous_ngrams(self):
        all_ngrams = Counter({"hello world": 2, "world test": 1})
        sentence_ngrams = ["hello world", "world test"]
        previous_ngrams = {"hello world"}
        expected = 2.5  # 1/2 + 1/1 + 1 (penalty) = 0.5 + 1.0 + 1.0
        result = get_ngram_weight(all_ngrams, sentence_ngrams, previous_ngrams)
        assert abs(
            result - expected) < 1e-6, f"Expected {expected}, but got {result}"


class TestSortSentences:
    def test_sort_sentences_basic(self):
        sentences = ["hello world", "world test", "hello test"]
        n = 1
        # Order may vary based on ngram weights
        expected = ["hello world", "world test", "hello test"]
        with patch("tqdm.tqdm", lambda x, **kwargs: x):  # Mock tqdm.tqdm
            result = sort_sentences(sentences, n)
        assert len(result) == len(
            expected), f"Expected length {len(expected)}, but got {len(result)}"
        assert set(result) == set(
            expected), f"Expected {expected}, but got {result}"


class TestNGramFrequency:
    def test_n_gram_frequency_bigrams(self):
        sentence = "hello world hello"
        n = 2
        expected = Counter({"he": 2, "el": 2, "ll": 2, "lo": 2, "o ": 1, " w": 1,
                           "wo": 1, "or": 1, "rl": 1, "ld": 1, "d ": 1, " h": 1})
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
        freq = Counter({"hello world": 1, "world test": 2})
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
        sentences = ["hello world test", "hello world again", "test case"]
        n = 2
        top_n = 2
        is_start_ngrams = True
        with patch("tqdm.tqdm", lambda x, **kwargs: x):
            result = filter_and_sort_sentences_by_ngrams(
                sentences, n, top_n, is_start_ngrams)
        expected = ["hello world test", "hello world again",
                    "test case"]  # All sentences may be included
        assert len(result) <= len(
            sentences), f"Expected length <= {len(sentences)}, but got {len(result)}"
        assert all(
            s in sentences for s in result), f"Result contains invalid sentences: {result}"


class TestStratifiedSampler:
    def test_init_with_float_num_samples(self):
        data = [ProcessedDataString(source="hello world", category_values=["q1"]),
                ProcessedDataString(source="test case", category_values=["q2"])]
        num_samples = 0.5
        sampler = StratifiedSampler(data, num_samples)
        expected = 1
        result = sampler.num_samples
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_init_with_invalid_num_samples(self):
        data = [ProcessedDataString(
            source="hello world", category_values=["q1"])]
        num_samples = 0.0
        expected = ValueError
        with pytest.raises(expected):
            StratifiedSampler(data, num_samples)

    def test_filter_strings(self):
        data = [ProcessedDataString(source="hello world test", category_values=["q1"]),
                ProcessedDataString(
                    source="hello world again", category_values=["q1"]),
                ProcessedDataString(source="test case", category_values=["q2"])]
        sampler = StratifiedSampler(data, num_samples=2)
        with patch("tqdm.tqdm", lambda x, **kwargs: x):
            result = sampler.filter_strings(n=2, top_n=2)
        expected = 2
        assert len(
            result) == expected, f"Expected length {expected}, but got {len(result)}"
        assert all(isinstance(s, str)
                   for s in result), "Result contains non-string items"

    def test_get_samples(self):
        data = [
            ProcessedData(source="hello", target="world",
                          category_values=["q1"], score=0.9),
            ProcessedData(source="hello again", target="world again",
                          category_values=["q1"], score=0.7),
            ProcessedData(source="test", target="case",
                          category_values=["q2"], score=0.8),
            ProcessedData(source="test again", target="case again",
                          category_values=["q2"], score=0.6)
        ]
        sampler = StratifiedSampler(data, num_samples=2)
        with patch("tqdm.tqdm", lambda x, **kwargs: x):
            result = sampler.get_samples()
        expected = 2
        assert len(
            result) == expected, f"Expected length {expected}, but got {len(result)}"
        assert all(isinstance(item, dict)
                   for item in result), "Result contains non-dict items"
        assert result[0]["source"] in ["hello", "hello again",
                                       "test", "test again"], f"Invalid source in result: {result}"

    def test_get_unique_strings(self):
        data = [ProcessedDataString(source="hello world", category_values=["q1"]),
                ProcessedDataString(source="test case", category_values=["q2"])]
        sampler = StratifiedSampler(data, num_samples=1)
        with patch("tqdm.tqdm", lambda x, **kwargs: x):
            result = sampler.get_unique_strings()
        expected = 1
        assert len(
            result) == expected, f"Expected length {expected}, but got {len(result)}"
        assert all(isinstance(s, str)
                   for s in result), "Result contains non-string items"

    def test_load_data_with_labels(self):
        data = ["hello world", "test case example"]
        sampler = StratifiedSampler(data, num_samples=1)
        with patch("tqdm.tqdm", lambda x, **kwargs: x):
            result = sampler.load_data_with_labels(max_q=2)
        expected = 2  # Expect full sentences, not split words
        assert len(
            result) == expected, f"Expected length {expected}, but got {len(result)}"
        assert all(isinstance(item, dict)
                   for item in result), "Result contains non-dict items"
        assert all(len(item['category_values']) ==
                   4 for item in result), "Category values length incorrect"
