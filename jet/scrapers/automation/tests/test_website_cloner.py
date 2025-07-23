import pytest

from jet.scrapers.automation.website_cloner import WebsiteCloner

# Tests using pytest with BDD-style structure


class TestWebsiteCloner:
    def test_fetch_website_success(self):
        # Given: A valid website URL
        url = "https://example.com"
        cloner = WebsiteCloner(url)

        # When: Fetching website content
        result = cloner.fetch_website()

        # Then: Content is fetched with non-empty HTML and title
        expected_title = "Example Domain"
        assert result.title == expected_title
        assert len(result.html) > 0
        assert isinstance(result.css, list)

    def test_generate_tailwind_html(self):
        # Given: A fetched website
        url = "https://example.com"
        cloner = WebsiteCloner(url)
        cloner.fetch_website()

        # When: Generating Tailwind HTML
        result = cloner.generate_tailwind_html()

        # Then: HTML contains Tailwind CDN and expected structure
        expected_elements = [
            '<!DOCTYPE html>',
            'tailwind.min.css',
            '<body>',
            'text-gray-800',
            'my-2'
        ]
        for element in expected_elements:
            assert element in result

    def test_fetch_website_invalid_url(self):
        # Given: An invalid URL
        url = "https://nonexistent.example.com"
        cloner = WebsiteCloner(url)

        # When: Attempting to fetch content
        # Then: Raises HTTP error
        with pytest.raises(requests.HTTPError):
            cloner.fetch_website()
