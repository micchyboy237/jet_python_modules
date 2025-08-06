from jet.scrapers.utils import scrape_links


class TestScrapeLinks:
    def test_absolute_urls(self):
        # Given: Text with multiple absolute URLs
        text = """
        Visit https://example.com for more info.
        Check out http://test.com/page1 and https://another.com/path/to/page
        """
        expected = [
            "https://example.com",
            "http://test.com/page1",
            "https://another.com/path/to/page"
        ]

        # When: We scrape links from the text
        result = scrape_links(text)

        # Then: All absolute URLs are extracted correctly
        assert result == expected

    def test_relative_paths_with_base_url(self):
        # Given: Text with relative paths and a base URL
        text = """
        Go to /page1 or /path/to/page2
        """
        base_url = "https://example.com"
        expected = [
            "https://example.com/page1",
            "https://example.com/path/to/page2"
        ]

        # When: We scrape links with a base URL
        result = scrape_links(text, base_url)

        # Then: Relative paths are converted to absolute URLs
        assert result == expected

    def test_mixed_urls_with_base_url(self):
        # Given: Text with both absolute URLs and relative paths
        text = """
        Absolute: https://example.com/page
        Relative: /path1 and /path/to/page2
        """
        base_url = "https://test.com"
        expected = [
            "https://example.com/page",
            "https://test.com/path1",
            "https://test.com/path/to/page2"
        ]

        # When: We scrape links with a base URL
        result = scrape_links(text, base_url)

        # Then: Both absolute and relative URLs are handled correctly
        assert result == expected

    def test_no_urls(self):
        # Given: Text with no URLs
        text = "This is plain text with no links"
        expected = []

        # When: We scrape links from text with no URLs
        result = scrape_links(text)

        # Then: An empty list is returned
        assert result == expected

    def test_invalid_urls(self):
        # Given: Text with invalid URLs
        text = """
        Invalid: ftp://example.com
        Malformed: http://[invalid
        """
        expected = []

        # When: We scrape links with a base URL
        result = scrape_links(text)

        # Then: Only valid URLs are returned
        assert result == expected

    def test_duplicate_urls(self):
        # Given: Text with duplicate URLs
        text = """
        https://example.com
        https://example.com
        /page1
        /page1
        """
        base_url = "https://test.com"
        expected = [
            "https://example.com",
            "https://test.com/page1"
        ]

        # When: We scrape links with duplicates
        result = scrape_links(text, base_url)

        # Then: Duplicates are removed
        assert result == expected

    def test_urls_with_query_params_and_fragments(self):
        # Given: Text with URLs containing query parameters and fragments
        text = """
        https://example.com/page?q=123#section1
        /path?page=2
        """
        base_url = "https://test.com"
        expected = [
            "https://example.com/page?q=123#section1",
            "https://test.com/path?page=2"
        ]

        # When: We scrape links with query params and fragments
        result = scrape_links(text, base_url)

        # Then: URLs with query params and fragments are preserved
        assert result == expected

    def test_empty_text(self):
        # Given: Empty text
        text = ""
        expected = []

        # When: We scrape links from empty text
        result = scrape_links(text)

        # Then: An empty list is returned
        assert result == expected

    def test_base_url_without_trailing_slash(self):
        # Given: Text with relative path and base URL without trailing slash
        text = "/page1"
        base_url = "https://example.com"
        expected = ["https://example.com/page1"]

        # When: We scrape links with base URL
        result = scrape_links(text, base_url)

        # Then: Relative path is correctly resolved
        assert result == expected

    def test_relative_paths_with_base_url(self):
        """
        Given: Text containing relative paths starting with '/', including query parameters and fragments
        When: scrape_links is called without a base_url
        Then: The relative paths are returned as-is, preserving query parameters and fragments
        """
        text = """
        Go to /page1, /page1?q=123 or /page1#section1
        """
        base_url = "https://example.com"
        expected = [
            "https://example.com/page1",
            "https://example.com/page1?q=123",
            "https://example.com/page1#section1"
        ]
        result = scrape_links(text, base_url)
        assert result == expected

    def test_relative_paths_without_base_url(self):
        """
        Given: Text containing relative paths starting with '/', including query parameters and fragments
        When: scrape_links is called without a base_url
        Then: The relative paths are returned as-is, preserving query parameters and fragments
        """
        text = """
        Go to /page1, /page1?q=123 or /page1#section1
        """
        expected = [
            "/page1",
            "/page1?q=123",
            "/page1#section1"
        ]
        result = scrape_links(text)
        assert result == expected
