from jet.scrapers.utils import filter_links


class TestFilterLinks:
    def test_filters_by_domain_pattern(self):
        # Given
        urls = [
            "https://example.com/home",
            "https://sub.example.com/about",
            "https://another.com/page",
        ]

        patterns = ["https://*.example.com/*"]

        # When
        result = filter_links(urls, patterns)

        # Then
        expected = [
            "https://sub.example.com/about",
        ]

        assert result == expected

    def test_filters_by_keyword_pattern(self):
        # Given
        urls = [
            "https://site.com/login",
            "https://site.com/register",
            "https://site.com/dashboard",
        ]

        patterns = ["*login*", "*register*"]

        # When
        result = filter_links(urls, patterns)

        # Then
        expected = [
            "https://site.com/login",
            "https://site.com/register",
        ]

        assert result == expected

    def test_returns_empty_when_no_match(self):
        # Given
        urls = [
            "https://site.com/a",
            "https://site.com/b",
        ]

        patterns = ["*login*"]

        # When
        result = filter_links(urls, patterns)

        # Then
        expected: list[str] = []

        assert result == expected

    def test_matches_multiple_patterns(self):
        # Given
        urls = [
            "https://docs.python.org",
            "https://pypi.org/project",
            "https://github.com/python/cpython",
        ]

        patterns = [
            "*python*",
            "*pypi*",
        ]

        # When
        result = filter_links(urls, patterns)

        # Then
        expected = [
            "https://docs.python.org",
            "https://pypi.org/project",
            "https://github.com/python/cpython",
        ]

        assert result == expected

    def test_exact_match_pattern(self):
        # Given
        urls = [
            "https://example.com/page",
            "https://example.com/other",
        ]

        patterns = ["https://example.com/page"]

        # When
        result = filter_links(urls, patterns)

        # Then
        expected = [
            "https://example.com/page",
        ]

        assert result == expected
