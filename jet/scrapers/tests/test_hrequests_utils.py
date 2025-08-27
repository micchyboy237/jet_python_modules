import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from jet.scrapers.hrequests_utils import scrape_url, scrape_urls, scrape_url_sync
from fake_useragent import UserAgent


class TestHRequestsUtils(unittest.TestCase):
    def setUp(self):
        self.ua = UserAgent()
        self.mock_session = AsyncMock()
        self.urls = [
            "https://example.com",
            "https://python.org",
            "https://github.com",
            "https://httpbin.org/html",
            "https://wikipedia.org",
            "https://invalid-url.com",
            "https://timeout.com",
            "https://another.com",
            "https://test.com",
            "https://extra.com",
            "https://more.com",
            "https://final.com",
        ]

    @pytest.mark.asyncio
    async def test_scrape_url_success(self):
        # Mock successful response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(
            return_value="<html>Test content</html>")
        self.mock_session.get.return_value.__aenter__.return_value = mock_response

        with patch("jet.scrapers.hrequests_utils.cache.get", return_value=None):
            with patch("jet.scrapers.hrequests_utils.cache.set") as mock_cache_set:
                with patch("jet.scrapers.hrequests_utils.logger") as mock_logger:
                    result = await scrape_url(self.mock_session, "https://example.com", self.ua, timeout=5.0)
                    self.assertEqual(result, "<html>Test content</html>")
                    mock_cache_set.assert_called_once_with(
                        "html:https://example.com",
                        {"content": "<html>Test content</html>"},
                        ttl=3600
                    )
                    mock_logger.error.assert_not_called()

    @pytest.mark.asyncio
    async def test_scrape_url_cached(self):
        # Mock cached content
        with patch("jet.scrapers.hrequests_utils.cache.get", return_value={"content": "<html>Cached content</html>"}):
            with patch("jet.scrapers.hrequests_utils.cache.set") as mock_cache_set:
                result = await scrape_url(self.mock_session, "https://example.com", self.ua, timeout=5.0)
                self.assertEqual(result, "<html>Cached content</html>")
                self.mock_session.get.assert_not_called()
                mock_cache_set.assert_not_called()

    @pytest.mark.asyncio
    async def test_scrape_url_timeout(self):
        # Mock timeout
        self.mock_session.get.side_effect = asyncio.TimeoutError
        with patch("jet.scrapers.hrequests_utils.cache.get", return_value=None):
            with patch("jet.scrapers.hrequests_utils.logger") as mock_logger:
                result = await scrape_url(self.mock_session, "https://timeout.com", self.ua, timeout=1.0)
                self.assertIsNone(result)
                mock_logger.error.assert_called_once_with(
                    "Timeout fetching https://timeout.com: Exceeded 1.0 seconds")

    @pytest.mark.asyncio
    async def test_scrape_url_failure(self):
        # Mock failed response
        mock_response = AsyncMock()
        mock_response.status = 404
        mock_response.reason = "Not Found"
        self.mock_session.get.return_value.__aenter__.return_value = mock_response

        with patch("jet.scrapers.hrequests_utils.cache.get", return_value=None):
            with patch("jet.scrapers.hrequests_utils.logger") as mock_logger:
                result = await scrape_url(self.mock_session, "https://invalid-url.com", self.ua, timeout=5.0)
                self.assertIsNone(result)
                mock_logger.warning.assert_called_once_with(
                    "Failed: https://invalid-url.com - Status Code: 404, Reason: Not Found"
                )

    @pytest.mark.asyncio
    async def test_scrape_urls_basic(self):
        # Mock responses for 5 URLs
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="<html>Test</html>")
        self.mock_session.get.return_value.__aenter__.return_value = mock_response

        with patch("aiohttp.ClientSession", return_value=self.mock_session):
            with patch("jet.scrapers.hrequests_utils.cache.get", return_value=None):
                with patch("jet.scrapers.hrequests_utils.cache.set"):
                    results = []
                    async for url, status, html in scrape_urls(self.urls[:5], num_parallel=3, limit=5, show_progress=False):
                        results.append((url, status, html))

        # Expect 10 results (5 URLs x 2 statuses: "started" and "completed")
        self.assertEqual(len(results), 10)
        completed = [r for r in results if r[1] == "completed"]
        self.assertEqual(len(completed), 5)
        for _, status, html in completed:
            self.assertEqual(status, "completed")
            self.assertEqual(html, "<html>Test</html>")

    @pytest.mark.asyncio
    async def test_scrape_urls_urls_double_limit(self):
        # Test with URLs > 2x limit (12 URLs, limit=5)
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="<html>Test</html>")
        self.mock_session.get.return_value.__aenter__.return_value = mock_response

        with patch("aiohttp.ClientSession", return_value=self.mock_session):
            with patch("jet.scrapers.hrequests_utils.cache.get", return_value=None):
                with patch("jet.scrapers.hrequests_utils.cache.set"):
                    results = []
                    async for url, status, html in scrape_urls(self.urls, num_parallel=3, limit=5, show_progress=False):
                        results.append((url, status, html))

        # Expect 10 results (5 URLs processed due to limit)
        self.assertEqual(len(results), 10)
        processed_urls = {r[0] for r in results}
        self.assertEqual(len(processed_urls), 5)
        self.assertTrue(all(url in self.urls[:5] for url in processed_urls))

    @pytest.mark.asyncio
    async def test_scrape_urls_limit_triple_num_parallel(self):
        # Test with limit > 3x num_parallel (limit=10, num_parallel=3)
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="<html>Test</html>")
        self.mock_session.get.return_value.__aenter__.return_value = mock_response

        with patch("aiohttp.ClientSession", return_value=self.mock_session):
            with patch("jet.scrapers.hrequests_utils.cache.get", return_value=None):
                with patch("jet.scrapers.hrequests_utils.cache.set"):
                    results = []
                    async for url, status, html in scrape_urls(self.urls, num_parallel=3, limit=10, show_progress=False):
                        results.append((url, status, html))

        # Expect 20 results (10 URLs x 2 statuses)
        self.assertEqual(len(results), 20)
        processed_urls = {r[0] for r in results}
        self.assertEqual(len(processed_urls), 10)
        self.assertTrue(all(url in self.urls[:10] for url in processed_urls))
        completed = [r for r in results if r[1] == "completed"]
        self.assertEqual(len(completed), 10)
        # Verify parallel execution: tasks should not exceed num_parallel
        active_tasks = 0
        max_active = 0
        for url, status, _ in results:
            if status == "started":
                active_tasks += 1
                max_active = max(max_active, active_tasks)
            elif status == "completed":
                active_tasks -= 1
        self.assertLessEqual(max_active, 3)

    @pytest.mark.asyncio
    async def test_scrape_urls_empty_list(self):
        # Test with empty URLousine URLs list
        with patch("aiohttp.ClientSession", return_value=self.mock_session):
            results = []
            async for url, status, html in scrape_urls([], num_parallel=3, limit=None, show_progress=False):
                results.append((url, status, html))
        self.assertEqual(len(results), 0)

    @pytest.mark.asyncio
    async def test_scrape_urls_with_blocking(self):
        # Test with some URLs blocking (simulated by TimeoutError)
        def get_side_effect(url, headers, timeout):
            if url in ["https://timeout.com", "https://invalid-url.com"]:
                raise asyncio.TimeoutError
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.text = AsyncMock(return_value="<html>Test</html>")
            return mock_response

        self.mock_session.get.side_effect = get_side_effect

        with patch("aiohttp.ClientSession", return_value=self.mock_session):
            with patch("jet.scrapers.hrequests_utils.cache.get", return_value=None):
                with patch("jet.scrapers.hrequests_utils.cache.set"):
                    with patch("jet.scrapers.hrequests_utils.logger"):
                        results = []
                        async for url, status, html in scrape_urls(self.urls[:5], num_parallel=2, limit=None, show_progress=False):
                            results.append((url, status, html))

        # Expect 10 results (5 URLs x 2 statuses), with 2 timeouts
        self.assertEqual(len(results), 10)
        completed = [r for r in results if r[1] == "completed"]
        self.assertEqual(len(completed), 5)
        timeout_results = [r for r in results if r[0] in [
            "https://timeout.com", "https://invalid-url.com"] and r[1] == "completed"]
        for url, _, html in timeout_results:
            self.assertIsNone(html)

    def test_sync_scrape_url_success(self):
        # Mock successful sync response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html>Sync Test</html>"
        with patch("requests.get", return_value=mock_response):
            with patch("jet.scrapers.hrequests_utils.cache.get", return_value=None):
                with patch("jet.scrapers.hrequests_utils.cache.set") as mock_cache_set:
                    result = scrape_url_sync("https://example.com")
                    self.assertEqual(result, "<html>Sync Test</html>")
                    mock_cache_set.assert_called_once_with(
                        "html:https://example.com",
                        {"content": "<html>Sync Test</html>"},
                        ttl=3600
                    )

    def test_sync_scrape_url_failure(self):
        # Mock failed sync response
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.reason = "Not Found"
        with patch("requests.get", return_value=mock_response):
            with patch("jet.scrapers.hrequests_utils.cache.get", return_value=None):
                with patch("jet.scrapers.hrequests_utils.logger") as mock_logger:
                    result = scrape_url_sync("https://invalid-url.com")
                    self.assertIsNone(result)
                    mock_logger.warning.assert_called_once_with(
                        "Failed: https://invalid-url.com - Status Code: 404, Reason: Not Found"
                    )

    @pytest.mark.asyncio
    async def test_scrape_urls_limit_based_on_completed(self):
        # Given: A mix of successful and failed URLs
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="<html>Test</html>")

        def get_side_effect(url, headers, timeout):
            if url in ["https://invalid-url.com", "https://timeout.com"]:
                return AsyncMock(status=404, reason="Not Found")
            return mock_response

        self.mock_session.get.side_effect = get_side_effect

        expected_urls = [
            "https://example.com",
            "https://python.org",
            "https://github.com",
        ]
        expected_results = [
            (url, "started", None) for url in expected_urls
        ] + [
            (url, "completed", "<html>Test</html>") for url in expected_urls
        ]

        # When: Scraping URLs with a limit of 3 completed responses
        with patch("aiohttp.ClientSession", return_value=self.mock_session):
            with patch("jet.scrapers.hrequests_utils.cache.get", return_value=None):
                with patch("jet.scrapers.hrequests_utils.cache.set"):
                    with patch("jet.scrapers.hrequests_utils.logger"):
                        results = []
                        async for url, status, html in scrape_urls(self.urls, num_parallel=3, limit=3, show_progress=False):
                            results.append((url, status, html))

        # Then: Verify only 3 successful completions are processed
        self.assertEqual(len(results), len(expected_results)
                         )  # 3 started + 3 completed
        completed = [r for r in results if r[1]
                     == "completed" and r[2] is not None]
        self.assertEqual(len(completed), 3)
        self.assertEqual(results, expected_results)
        processed_urls = {r[0] for r in results}
        self.assertEqual(processed_urls, set(expected_urls))


if __name__ == "__main__":
    unittest.main()
