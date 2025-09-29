import pytest
import requests
from unittest.mock import patch
from jet.scrapers.utils import extract_favicon_ico_link

@pytest.fixture
def mock_requests_get():
    with patch('requests.get') as mock_get:
        yield mock_get

@pytest.fixture
def mock_requests_head():
    with patch('requests.head') as mock_head:
        yield mock_head

class TestExtractFaviconIcoLink:
    def test_extracts_favicon_from_link_tag_url(self, mock_requests_get):
        # Given: A website with a favicon specified in a <link> tag
        html_content = '''
        <html>
            <head>
                <link rel="shortcut icon" href="/images/favicon.ico">
            </head>
        </html>
        '''
        mock_requests_get.return_value.text = html_content
        mock_requests_get.return_value.status_code = 200
        test_source = "https://example.com"
        expected = "https://example.com/images/favicon.ico"
        
        # When: We call the function with the URL
        result = extract_favicon_ico_link(test_source)
        
        # Then: It returns the absolute favicon URL
        assert result == expected, f"Expected favicon URL {expected}, got {result}"

    def test_extracts_favicon_from_link_tag_html(self):
        # Given: An HTML string with a favicon specified in a <link> tag
        html_content = '''
        <html>
            <head>
                <link rel="shortcut icon" href="/images/favicon.ico">
            </head>
        </html>
        '''
        test_source = html_content
        expected = "https://example.com/images/favicon.ico"
        
        # When: We call the function with the HTML string
        result = extract_favicon_ico_link(test_source)
        
        # Then: It returns the absolute favicon URL
        assert result == expected, f"Expected favicon URL {expected}, got {result}"

    def test_fallback_to_default_favicon_url(self, mock_requests_get, mock_requests_head):
        # Given: A website without favicon in <link> tag but with default favicon.ico
        html_content = '<html><head></head></html>'
        mock_requests_get.return_value.text = html_content
        mock_requests_get.return_value.status_code = 200
        mock_requests_head.return_value.status_code = 200
        test_source = "https://example.com"
        expected = "https://example.com/favicon.ico"
        
        # When: We call the function with the URL
        result = extract_favicon_ico_link(test_source)
        
        # Then: It returns the default favicon URL
        assert result == expected, f"Expected default favicon URL {expected}, got {result}"

    def test_no_fallback_for_html_input(self):
        # Given: An HTML string without favicon in <link> tag
        html_content = '<html><head></head></html>'
        test_source = html_content
        expected = None
        
        # When: We call the function with the HTML string
        result = extract_favicon_ico_link(test_source)
        
        # Then: It returns None since no default favicon check for HTML
        assert result == expected, f"Expected None for HTML input without favicon, got {result}"

    def test_returns_none_on_invalid_url(self, mock_requests_get):
        # Given: An invalid URL that raises a request exception
        mock_requests_get.side_effect = requests.RequestException
        test_source = "https://invalid-url"
        expected = None
        
        # When: We call the function with an invalid URL
        result = extract_favicon_ico_link(test_source)
        
        # Then: It returns None
        assert result == expected, f"Expected None for invalid URL, got {result}"

    def test_returns_none_when_no_favicon_found_url(self, mock_requests_get, mock_requests_head):
        # Given: A website with no favicon in <link> tag and no default favicon
        html_content = '<html><head></head></html>'
        mock_requests_get.return_value.text = html_content
        mock_requests_get.return_value.status_code = 200
        mock_requests_head.return_value.status_code = 404
        test_source = "https://example.com"
        expected = None
        
        # When: We call the function with the URL
        result = extract_favicon_ico_link(test_source)
        
        # Then: It returns None
        assert result == expected, f"Expected None when no favicon found, got {result}"

