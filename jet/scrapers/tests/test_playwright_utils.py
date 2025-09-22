import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Optional, Tuple
from jet.scrapers.playwright_utils import scrape_url, scrape_urls, scrape_urls_sync, ScrapeStatus
from jet.cache.redis.utils import RedisCache

@pytest.fixture
def mock_playwright():
    with patch("jet.scrapers.playwright_utils.async_playwright") as mock_pw:
        mock_browser = AsyncMock()
        mock_context = AsyncMock()
        mock_page = AsyncMock()

        mock_pw.return_value.__aenter__.return_value = MagicMock(chromium=MagicMock(launch=AsyncMock(return_value=mock_browser)))
        mock_browser.new_context.return_value = mock_context
        mock_context.new_page.return_value = mock_page

        yield {
            "playwright": mock_pw,
            "browser": mock_browser,
            "context": mock_context,
            "page": mock_page
        }

@pytest.fixture
def mock_redis_cache():
    with patch("jet.scrapers.playwright_utils.cache", autospec=True) as mock_cache:
        yield mock_cache

class TestPlaywrightUtils:
    @pytest.mark.asyncio
    async def test_scrape_url_success_with_screenshot(self, mock_playwright, mock_redis_cache):
        # Given: A valid URL and mocked Playwright setup
        url = "https://python.org"
        expected_html = "<html><body>Test content</body></html>"
        expected_screenshot = b"mock_screenshot_data"
        mock_redis_cache.get.return_value = None
        mock_playwright["page"].goto = AsyncMock()
        mock_playwright["page"].content.return_value = expected_html
        mock_playwright["page"].screenshot.return_value = expected_screenshot

        # When: scrape_url is called with screenshot enabled
        result_html, result_screenshot = await scrape_url(
            context=mock_playwright["context"],
            url=url,
            timeout=5000,
            max_retries=2,
            with_screenshot=True
        )

        # Then: The correct HTML and screenshot are returned, and cached
        assert result_html == expected_html
        assert result_screenshot == expected_screenshot
        mock_redis_cache.set.assert_called_once_with(
            f"html:{url}", {"content": expected_html, "screenshot": expected_screenshot}, ttl=3600
        )

    @pytest.mark.asyncio
    async def test_scrape_url_cached_content(self, mock_playwright, mock_redis_cache):
        # Given: A URL with cached content
        url = "https://python.org"
        expected_html = "<html><body>Cached content</body></html>"
        expected_screenshot = b"cached_screenshot"
        mock_redis_cache.get.return_value = {"content": expected_html, "screenshot": expected_screenshot}

        # When: scrape_url is called
        result_html, result_screenshot = await scrape_url(
            context=mock_playwright["context"],
            url=url,
            timeout=5000,
            max_retries=2,
            with_screenshot=True
        )

        # Then: Cached content is returned without calling Playwright
        assert result_html == expected_html
        assert result_screenshot == expected_screenshot
        mock_playwright["page"].goto.assert_not_called()

    @pytest.mark.asyncio
    async def test_scrape_url_failure_with_retries(self, mock_playwright, mock_redis_cache):
        # Given: A URL that fails to load, with retries enabled
        url = "https://python.org"
        mock_redis_cache.get.return_value = None
        mock_playwright["page"].goto.side_effect = Exception("Network error")

        # When: scrape_url is called with max_retries=2
        result_html, result_screenshot = await scrape_url(
            context=mock_playwright["context"],
            url=url,
            timeout=5000,
            max_retries=2,
            with_screenshot=True
        )

        # Then: None is returned after retries, and no screenshot is taken
        assert result_html is None
        assert result_screenshot is None
        assert mock_playwright["page"].goto.call_count == 3  # Initial + 2 retries
        mock_redis_cache.set.assert_not_called()

    @pytest.mark.asyncio
    async def test_scrape_urls_multiple_urls(self, mock_playwright, mock_redis_cache):
        # Given: A list of URLs to scrape
        urls = ["https://python.org", "https://github.com"]
        expected_results = [
            (urls[0], "completed", "<html>Python</html>", b"screenshot1"),
            (urls[1], "completed", "<html>Github</html>", b"screenshot2")
        ]
        mock_redis_cache.get.return_value = None
        mock_playwright["page"].goto = AsyncMock()
        mock_playwright["page"].content.side_effect = [expected_results[0][2], expected_results[1][2]]
        mock_playwright["page"].screenshot.side_effect = [expected_results[0][3], expected_results[1][3]]

        # When: scrape_urls is called with multiple URLs
        results = []
        async for url, status, html, screenshot in scrape_urls(
            urls=urls,
            num_parallel=2,
            limit=None,
            show_progress=False,
            timeout=5000,
            max_retries=2,
            with_screenshot=True,
            headless=True
        ):
            if status != "started":
                results.append((url, status, html, screenshot))

        # Then: All URLs are scraped with correct results
        assert results == expected_results
        assert mock_redis_cache.set.call_count == 2

    def test_scrape_urls_sync(self, mock_playwright, mock_redis_cache):
        # Given: A list of URLs to scrape synchronously
        urls = ["https://python.org"]
        expected_results = [(urls[0], "completed", "<html>Python</html>", b"screenshot1")]
        mock_redis_cache.get.return_value = None
        mock_playwright["page"].goto = AsyncMock()
        mock_playwright["page"].content.return_value = expected_results[0][2]
        mock_playwright["page"].screenshot.return_value = expected_results[0][3]

        # When: scrape_urls_sync is called
        results = scrape_urls_sync(
            urls=urls,
            num_parallel=1,
            limit=None,
            show_progress=False,
            timeout=5000,
            max_retries=2,
            with_screenshot=True,
            headless=True
        )

        # Then: The correct results are returned
        assert results == [
            (urls[0], "started", None, None),
            expected_results[0]
        ]
        mock_redis_cache.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_scrape_urls_with_limit(self, mock_playwright, mock_redis_cache):
        # Given: A list of URLs with a limit of 1 successful scrape
        urls = ["https://python.org", "https://github.com"]
        expected_results = [(urls[0], "completed", "<html>Python</html>", b"screenshot1")]
        mock_redis_cache.get.return_value = None
        mock_playwright["page"].goto = AsyncMock()
        mock_playwright["page"].content.return_value = expected_results[0][2]
        mock_playwright["page"].screenshot.return_value = expected_results[0][3]

        # When: scrape_urls is called with limit=1
        results = []
        async for url, status, html, screenshot in scrape_urls(
            urls=urls,
            num_parallel=2,
            limit=1,
            show_progress=False,
            timeout=5000,
            max_retries=2,
            with_screenshot=True,
            headless=True
        ):
            if status != "started":
                results.append((url, status, html, screenshot))

        # Then: Only one URL is scraped due to the limit
        assert results == expected_results
        assert mock_playwright["page"].goto.call_count == 1
        mock_redis_cache.set.assert_called_once()