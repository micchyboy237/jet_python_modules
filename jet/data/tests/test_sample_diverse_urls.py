import pytest
from typing import List
from unittest.mock import patch
from jet.data.sample_diverse_urls import sample_diverse_urls


class TestSampleDiverseUrls:
    def test_sample_diverse_urls_basic(self):
        urls = [
            "https://example.com/page1",
            "https://example.com/page2",
            "https://test.com/different/path",
            "http://another.org/page"
        ]
        num_samples = 2
        n = 2
        top_n = 1
        expected = [
            "https://example.com/page1",
            "http://another.org/page"
        ]
        with patch("tqdm.tqdm", lambda x, **kwargs: x):
            result = sample_diverse_urls(urls, num_samples, n, top_n)
        assert sorted(result) == sorted(
            expected), f"Expected {expected}, but got {result}"

    def test_sample_diverse_urls_empty_list(self):
        urls = []
        num_samples = 2
        n = 2
        top_n = 1
        expected = []
        with patch("tqdm.tqdm", lambda x, **kwargs: x):
            result = sample_diverse_urls(urls, num_samples, n, top_n)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_sample_diverse_urls_single_url(self):
        urls = ["https://example.com/page"]
        num_samples = 1
        n = 2
        top_n = 1
        expected = ["https://example.com/page"]
        with patch("tqdm.tqdm", lambda x, **kwargs: x):
            result = sample_diverse_urls(urls, num_samples, n, top_n)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_sample_diverse_urls_more_samples_than_urls(self):
        urls = ["https://example.com/page1", "https://example.com/page2"]
        num_samples = 5
        n = 2
        top_n = 1
        expected = ["https://example.com/page1", "https://example.com/page2"]
        with patch("tqdm.tqdm", lambda x, **kwargs: x):
            result = sample_diverse_urls(urls, num_samples, n, top_n)
        assert result == expected, f"Expected {expected}, but got {result}"
        assert all(
            url in urls for url in result), f"Result contains invalid URLs: {result}"

    def test_sample_diverse_urls_duplicate_urls(self):
        urls = [
            "https://example.com/page",
            "https://example.com/page",
            "https://test.com/different"
        ]
        num_samples = 2
        n = 2
        top_n = 1
        expected = ["https://example.com/page", "https://test.com/different"]
        with patch("tqdm.tqdm", lambda x, **kwargs: x):
            result = sample_diverse_urls(urls, num_samples, n, top_n)
        assert sorted(result) == sorted(
            expected), f"Expected {expected}, but got {result}"
        assert len(set(result)) == len(
            result), f"Result contains duplicates: {result}"
        assert all(
            url in urls for url in result), f"Result contains invalid URLs: {result}"

    def test_sample_diverse_urls_with_fragments(self):
        urls = [
            "https://example.com/page1#fragment",
            "https://example.com/page2",
            "https://test.com/different/path"
        ]
        num_samples = 2
        n = 2
        top_n = 1
        expected = [
            "https://test.com/different/path",
            "https://example.com/page1#fragment"
        ]
        with patch("tqdm.tqdm", lambda x, **kwargs: x):
            result = sample_diverse_urls(urls, num_samples, n, top_n)
        assert sorted(result) == sorted(
            expected), f"Expected {expected}, but got {result}"
        assert all(
            url in urls for url in result), f"Result contains invalid URLs: {result}"

    def test_sample_diverse_urls_with_category_values(self):
        urls = [
            "http://example.com/page1",
            "http://example.com/page2",
            "http://test.com/page3"
        ]
        category_values = [
            ["ttr_q1", "q1", "ngram_q1", "start_ngram_q1"],
            ["ttr_q2", "q2", "ngram_q2", "start_ngram_q2"],
            ["ttr_q1", "q1", "ngram_q1", "start_ngram_q1"]
        ]
        num_samples = 2
        expected = [
            "http://example.com/page1",
            "http://example.com/page2"
        ]
        with patch("tqdm.tqdm", lambda x, **kwargs: x):
            result = sample_diverse_urls(
                urls, num_samples=num_samples, category_values=category_values)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_sample_diverse_urls_invalid_category_values_length(self):
        urls = ["http://example.com/page1", "http://example.com/page2"]
        category_values = [["ttr_q1", "q1"]]
        expected_error = "Length of category_values must match length of urls"
        with pytest.raises(ValueError) as exc_info:
            sample_diverse_urls(urls, category_values=category_values)
        result = str(exc_info.value)
        assert result == expected_error, f"Expected error {expected_error}, but got {result}"

    def test_sample_diverse_urls_category_values_different_categories(self):
        urls = [
            "http://example.com/news",
            "http://example.com/blog",
            "http://test.com/articles",
            "http://another.com/posts"
        ]
        category_values = [
            ["ttr_q1", "q1", "ngram_q1", "start_ngram_q1"],
            ["ttr_q2", "q2", "ngram_q2", "start_ngram_q2"],
            ["ttr_q1", "q3", "ngram_q1", "start_ngram_q1"],
            ["ttr_q2", "q1", "ngram_q3", "start_ngram_q3"]
        ]
        num_samples = 3
        expected = [
            "http://example.com/news",
            "http://example.com/blog",
            "http://test.com/articles"
        ]
        with patch("tqdm.tqdm", lambda x, **kwargs: x):
            result = sample_diverse_urls(
                urls, num_samples=num_samples, category_values=category_values)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_sample_diverse_urls_category_values_all_same(self):
        urls = [
            "http://example.com/page1",
            "http://example.com/page2",
            "http://test.com/page3"
        ]
        category_values = [
            ["ttr_q1", "q1", "ngram_q1", "start_ngram_q1"],
            ["ttr_q1", "q1", "ngram_q1", "start_ngram_q1"],
            ["ttr_q1", "q1", "ngram_q1", "start_ngram_q1"]
        ]
        num_samples = 2
        expected = [
            "http://example.com/page1",
            "http://example.com/page2"
        ]
        with patch("tqdm.tqdm", lambda x, **kwargs: x):
            result = sample_diverse_urls(
                urls, num_samples=num_samples, category_values=category_values)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_sample_diverse_urls_category_values_empty(self):
        urls = [
            "http://example.com/page1",
            "http://example.com/page2"
        ]
        category_values = [[], []]
        expected_error = "Length of category_values must match length of urls"
        with pytest.raises(ValueError) as exc_info:
            sample_diverse_urls(urls, category_values=category_values)
        result = str(exc_info.value)
        assert result == expected_error, f"Expected error {expected_error}, but got {result}"
