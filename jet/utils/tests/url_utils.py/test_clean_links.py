
from jet.utils.url_utils import clean_links


class TestCleanLinks:
    def test_clean_url_with_query_params(self):
        # Given: a text string containing a URL with query parameters
        input_text = "Visit https://example.com/path/page?param1=value1&param2=value2 for details"
        expected = "Visit https://example.com/path/page for details"
        
        # When: cleaning the links in the text
        result = clean_links(input_text)
        
        # Then: query parameters should be removed
        assert result == expected

    def test_clean_url_with_hash(self):
        # Given: a text string containing a URL with a hash fragment
        input_text = "See https://example.com/page#section1 for more info"
        expected = "See https://example.com/page for more info"
        
        # When: cleaning the links in the text
        result = clean_links(input_text)
        
        # Then: hash fragment should be removed
        assert result == expected

    def test_clean_url_with_trailing_slash(self):
        # Given: a text string containing a URL with a trailing slash
        input_text = "Check out https://example.com/path/ now"
        expected = "Check out https://example.com/path now"
        
        # When: cleaning the links in the text
        result = clean_links(input_text)
        
        # Then: trailing slash should be removed
        assert result == expected

    def test_clean_multiple_urls(self):
        # Given: a text string with multiple URLs
        input_text = "Links: https://site1.com/page1?x=1 and https://site2.com/path/#anchor"
        expected = "Links: https://site1.com/page1 and https://site2.com/path"
        
        # When: cleaning the links in the text
        result = clean_links(input_text)
        
        # Then: all URLs should be cleaned
        assert result == expected

    def test_no_urls_in_text(self):
        # Given: a text string without any URLs
        input_text = "This is plain text without links"
        expected = "This is plain text without links"
        
        # When: cleaning the links in the text
        result = clean_links(input_text)
        
        # Then: text should remain unchanged
        assert result == expected

    def test_invalid_url(self):
        # Given: a text string with an invalid URL
        input_text = "Try https://invalid_url for info"
        expected = "Try https://invalid_url for info"
        
        # When: cleaning the links in the text
        result = clean_links(input_text)
        
        # Then: invalid URL should remain unchanged
        assert result == expected
