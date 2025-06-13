from jet.utils.url_utils import parse_url, clean_url
import pytest


class TestParseUrl:
    def test_parse_url_full(self):
        url = "https://example.com/path/to/page?key1=value1&key2=value2#fragment"
        expected = ["https", "example", "com", "path", "to",
                    "page", "key1", "value1", "key2", "value2", "fragment"]
        result = parse_url(url)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_parse_url_no_query(self):
        url = "http://test.org/resource#fragment"
        expected = ["http", "test", "org", "resource", "fragment"]
        result = parse_url(url)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_parse_url_trailing_slash(self):
        url = "https://example.com/path/"
        expected = ["https", "example", "com", "path"]
        result = parse_url(url)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_parse_url_empty_path(self):
        url = "https://example.com/"
        expected = ["https", "example", "com"]
        result = parse_url(url)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_parse_url_no_scheme(self):
        url = "example.com/path"
        expected = ["https", "example", "com", "path"]
        result = parse_url(url)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_parse_url_only_domain(self):
        url = "example.com"
        expected = ["https", "example", "com"]
        result = parse_url(url)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_parse_url_multiple_query_values(self):
        url = "https://example.com/path?key1=value1&key1=value2"
        expected = ["https", "example", "com", "path",
                    "key1", "value1", "key1", "value2"]
        result = parse_url(url)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_parse_url_duplicate_query_values(self):
        url = "https://example.com/path?key1=value1&key1=value1"
        expected = ["https", "example", "com", "path",
                    "key1", "value1", "key1", "value1"]
        result = parse_url(url)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_parse_url_empty_query(self):
        url = "https://example.com/path?"
        expected = ["https", "example", "com", "path"]
        result = parse_url(url)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_parse_url_only_fragment(self):
        url = "https://example.com#fragment"
        expected = ["https", "example", "com", "fragment"]
        result = parse_url(url)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_parse_url_empty(self):
        url = ""
        expected = []
        result = parse_url(url)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_parse_url_hyphens_underscores(self):
        url = "https://example.com/path-to_resource?key_1=value-1#frag_ment"
        expected = ["https", "example", "com", "path to resource",
                    "key 1", "value 1", "frag ment"]
        result = parse_url(url)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_parse_url_subdomain(self):
        url = "https://sub.example.com/path"
        expected = ["https", "sub", "example", "com", "path"]
        result = parse_url(url)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_parse_url_with_port(self):
        url = "https://example.com:8080/path"
        expected = ["https", "example", "com", "path"]
        result = parse_url(url)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_parse_url_query_without_value(self):
        url = "https://example.com/path?key1=&key2"
        expected = ["https", "example", "com", "path", "key1", "key2"]
        result = parse_url(url)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_parse_url_multiple_fragments(self):
        url = "https://example.com/path#fragment1#fragment2"
        expected = ["https", "example", "com", "path", "fragment1#fragment2"]
        result = parse_url(url)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_parse_url_invalid_url(self):
        url = "http://[invalid"
        with pytest.raises(ValueError):
            result = parse_url(url)

    def test_parse_url_trailing_hash(self):
        url = "https://example.com/path#"
        expected = ["https", "example", "com", "path"]
        result = parse_url(url)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_parse_url_non_ascii(self):
        url = "https://example.com/über?key=värld#frägment"
        expected = ["https", "example", "com",
                    "uber", "key", "varld", "fragment"]
        result = parse_url(url)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_parse_url_encoded_text(self):
        url = "https://example.com/%C3%BCber/path%20with%20spaces?key%3D%C3%A4rld#fr%C3%A4gment"
        expected = ["https", "example", "com", "uber",
                    "path with spaces", "key", "arld", "fragment"]
        result = parse_url(url)
        assert result == expected, f"Expected {expected}, but got {result}"


class TestCleanUrl:
    def test_clean_url_no_scheme(self):
        url = "example.com/path"
        expected = "https://example.com/path"
        result = clean_url(url)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_clean_url_trailing_slash(self):
        url = "https://example.com/path/"
        expected = "https://example.com/path"
        result = clean_url(url)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_fragment(self):
        result = clean_url("https://example.com/path#héllo")
        expected = "https://example.com/path#héllo"
        assert result == expected, "Fragment should remain unchanged"

    def test_empty_fragment(self):
        result = clean_url("https://example.com/path#")
        expected = "https://example.com/path"
        assert result == expected, "Trailing hashtag should be removed"

    def test_clean_url_empty(self):
        url = ""
        expected = ""
        result = clean_url(url)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_clean_url_invalid_url(self):
        url = "http://[invalid"
        with pytest.raises(ValueError):
            result = clean_url(url)

    def test_clean_url_redundant_slashes(self):
        url = "https://example.com//path//to///page"
        expected = "https://example.com/path/to/page"
        result = clean_url(url)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_clean_url_mixed_case_scheme(self):
        url = "HTTPS://example.com/path"
        expected = "https://example.com/path"
        result = clean_url(url)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_clean_url_with_port(self):
        url = "https://example.com:8080/path"
        expected = "https://example.com:8080/path"
        result = clean_url(url)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_clean_url_trailing_hash(self):
        url = "https://example.com/path#"
        expected = "https://example.com/path"
        result = clean_url(url)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_clean_url_non_ascii(self):
        url = "https://example.com/über?key=värld#frägment"
        expected = "https://example.com/uber?key=varld#fragment"
        result = clean_url(url)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_clean_url_encoded_unicode(self):
        url = "https://example.com/%C3%BCber?key=%C3%A4rld#fr%C3%A4gment"
        expected = "https://example.com/uber?key=arld#fragment"
        result = clean_url(url)
        assert result == expected, f"Expected {expected}, but got {result}"
