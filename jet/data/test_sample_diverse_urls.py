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
