import pytest
from typing import List
from unittest.mock import patch
from jet.data.url_sampler import preprocess_url, sample_diverse_urls


class TestPreprocessUrl:
    def test_preprocess_url_full(self):
        url = "https://example.com/path/to/page?key1=value1&key2=value2#fragment"
        expected = ["https", "example", "com", "path", "to",
                    "page", "key1", "value1", "key2", "value2", "fragment"]
        result = preprocess_url(url)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_preprocess_url_no_query(self):
        url = "http://test.org/resource#fragment"
        expected = ["http", "test", "org", "resource", "fragment"]
        result = preprocess_url(url)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_preprocess_url_trailing_slash(self):
        url = "https://example.com/path/"
        expected = ["https", "example", "com", "path"]
        result = preprocess_url(url)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_preprocess_url_empty_path(self):
        url = "https://example.com/"
        expected = ["https", "example", "com"]
        result = preprocess_url(url)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_preprocess_url_no_scheme(self):
        url = "example.com/path"
        expected = ["https", "example", "com", "path"]
        result = preprocess_url(url)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_preprocess_url_only_domain(self):
        url = "example.com"
        expected = ["https", "example", "com"]
        result = preprocess_url(url)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_preprocess_url_multiple_query_values(self):
        url = "https://example.com/path?key1=value1&key1=value2"
        expected = ["https", "example", "com", "path",
                    "key1", "value1", "key1", "value2"]
        result = preprocess_url(url)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_preprocess_url_duplicate_query_values(self):
        url = "https://example.com/path?key1=value1&key1=value1"
        expected = ["https", "example", "com", "path",
                    "key1", "value1", "key1", "value1"]
        result = preprocess_url(url)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_preprocess_url_empty_query(self):
        url = "https://example.com/path?"
        expected = ["https", "example", "com", "path"]
        result = preprocess_url(url)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_preprocess_url_only_fragment(self):
        url = "https://example.com#fragment"
        expected = ["https", "example", "com", "fragment"]
        result = preprocess_url(url)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_preprocess_url_empty(self):
        url = ""
        expected = []
        result = preprocess_url(url)
        assert result == expected, f"Expected {expected}, but got {result}"


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
        with patch("tqdm.tqdm", lambda x, **kwargs: x):
            result = sample_diverse_urls(urls, num_samples, n, top_n)
        expected = 2
        assert len(
            result) == expected, f"Expected length {expected}, but got {len(result)}"
        assert all(
            url in urls for url in result), f"Result contains invalid URLs: {result}"

    def test_sample_diverse_urls_empty_list(self):
        urls = []
        num_samples = 2
        n = 2
        top_n = 1
        with patch("tqdm.tqdm", lambda x, **kwargs: x):
            result = sample_diverse_urls(urls, num_samples, n, top_n)
        expected = []
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_sample_diverse_urls_single_url(self):
        urls = ["https://example.com/page"]
        num_samples = 1
        n = 2
        top_n = 1
        with patch("tqdm.tqdm", lambda x, **kwargs: x):
            result = sample_diverse_urls(urls, num_samples, n, top_n)
        expected = ["https://example.com/page"]
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_sample_diverse_urls_more_samples_than_urls(self):
        urls = ["https://example.com/page1", "https://example.com/page2"]
        num_samples = 5
        n = 2
        top_n = 1
        with patch("tqdm.tqdm", lambda x, **kwargs: x):
            result = sample_diverse_urls(urls, num_samples, n, top_n)
        expected = 2
        assert len(
            result) <= expected, f"Expected max length {expected}, but got {len(result)}"
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
        with patch("tqdm.tqdm", lambda x, **kwargs: x):
            result = sample_diverse_urls(urls, num_samples, n, top_n)
        expected = 2
        assert len(
            result) == expected, f"Expected length {expected}, but got {len(result)}"
        assert len(set(result)) == len(
            result), f"Result contains duplicates: {result}"
        assert all(
            url in urls for url in result), f"Result contains invalid URLs: {result}"

    def test_sample_diverse_urls_with_fragments(self):
        urls = [
            "https://example.com/page1#section1",
            "https://example.com/page2#section2",
            "https://test.com/different/path"
        ]
        num_samples = 2
        n = 2
        top_n = 1
        with patch("tqdm.tqdm", lambda x, **kwargs: x):
            result = sample_diverse_urls(urls, num_samples, n, top_n)
        expected = 2
        assert len(
            result) == expected, f"Expected length {expected}, but got {len(result)}"
        assert all(
            url in urls for url in result), f"Result contains invalid URLs: {result}"
