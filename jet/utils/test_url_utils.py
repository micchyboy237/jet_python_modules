from jet.utils.url_utils import preprocess_url
import pytest


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

    def test_preprocess_url_hyphens_underscores(self):
        url = "https://example.com/path-to_resource?key_1=value-1#frag_ment"
        expected = ["https", "example", "com", "path to resource",
                    "key 1", "value 1", "frag ment"]
        result = preprocess_url(url)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_preprocess_url_subdomain(self):
        url = "https://sub.example.com/path"
        expected = ["https", "sub", "example", "com", "path"]
        result = preprocess_url(url)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_preprocess_url_with_port(self):
        url = "https://example.com:8080/path"
        expected = ["https", "example", "com", "path"]
        result = preprocess_url(url)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_preprocess_url_query_without_value(self):
        url = "https://example.com/path?key1=&key2"
        expected = ["https", "example", "com", "path", "key1", "key2"]
        result = preprocess_url(url)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_preprocess_url_non_ascii(self):
        url = "https://example.com/über?key=värld#frägment"
        expected = ["https", "example", "com",
                    "über", "key", "värld", "frägment"]
        result = preprocess_url(url)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_preprocess_url_multiple_fragments(self):
        url = "https://example.com/path#fragment1#fragment2"
        expected = ["https", "example", "com", "path", "fragment1#fragment2"]
        result = preprocess_url(url)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_preprocess_url_invalid_url(self):
        url = "http://[invalid"
        with pytest.raises(ValueError):
            result = preprocess_url(url)

    def test_preprocess_url_trailing_hash(self):
        url = "https://example.com/path#"
        expected = ["https", "example", "com", "path"]
        result = preprocess_url(url)
        assert result == expected, f"Expected {expected}, but got {result}"
