import pytest
from typing import List
from unittest.mock import patch
from jet.data.url_sampler import preprocess_url, sample_diverse_urls


class TestUrlSampler:
    def test_preprocess_url(self):
        url = "https://example.com/path/to/page?key1=value1&key2=value2"
        expected = ["https", "example", "com", "path", "to",
                    "page", "key1", "value1", "key2", "value2"]
        result = preprocess_url(url)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_preprocess_url_no_query(self):
        url = "http://test.org/resource"
        expected = ["http", "test", "org", "resource"]
        result = preprocess_url(url)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_sample_diverse_urls(self):
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
