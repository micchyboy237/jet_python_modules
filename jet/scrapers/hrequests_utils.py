import requests
import aiohttp
import asyncio
import sys
from fake_useragent import UserAgent
from typing import AsyncIterator, List, Optional, Tuple
from jet.code.splitter_markdown_utils import get_md_header_contents
from jet.logger import logger
from jet.scrapers.utils import scrape_links
from jet.cache.redis.types import RedisConfigParams
from jet.cache.redis.utils import RedisCache
from tqdm.asyncio import tqdm_asyncio  # Use tqdm_asyncio for async integration

REDIS_CONFIG = RedisConfigParams(
    port=3102
)
cache = RedisCache(config=REDIS_CONFIG)


def sync_scrape_url(url: str) -> Optional[str]:
    cache_key = f"html:{url}"
    cached_content = cache.get(cache_key)

    if cached_content:
        return cached_content['content']

    try:
        ua = UserAgent()
        headers = {'User-Agent': ua.random}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            html_content = response.text
            cache.set(cache_key, {'content': html_content}, ttl=3600)
            return html_content
        else:
            logger.warning(
                f"Failed: {url} - Status Code: {response.status_code}, Reason: {response.reason}")
            return None
    except Exception as e:
        logger.error(f"Error fetching {url}: {str(e)}")
        return None


def sync_scrape_urls(urls: List[str], num_parallel: int = 5) -> List[Optional[str]]:
    ua = UserAgent()
    results = []
    for url in urls:
        result = sync_scrape_url(url, ua)
        results.append(result)
    return results


async def scrape_url(session: aiohttp.ClientSession, url: str, ua: UserAgent, timeout: Optional[float] = 5.0) -> Optional[str]:
    """
    Scrape a single URL and return its HTML content, with caching and timeout handling.

    Args:
        session: aiohttp ClientSession for making HTTP requests.
        url: The URL to scrape.
        ua: UserAgent instance for generating random User-Agent headers.
        timeout: Optional timeout in seconds for the HTTP request (default: 5.0, None for no timeout).

    Returns:
        The HTML content as a string, or None if the request fails or times out.
    """
    cache_key = f"html:{url}"
    cached_content = cache.get(cache_key)

    if cached_content:
        return cached_content['content']

    try:
        headers = {'User-Agent': ua.random}
        client_timeout = aiohttp.ClientTimeout(
            total=timeout) if timeout is not None else None
        async with session.get(url, headers=headers, timeout=client_timeout) as response:
            if response.status == 200:
                html_content = await response.text()
                cache.set(cache_key, {'content': html_content}, ttl=3600)
                return html_content
            else:
                logger.warning(
                    f"Failed: {url} - Status Code: {response.status}, Reason: {response.reason}")
                return None
    except asyncio.TimeoutError:
        logger.error(f"Timeout fetching {url}: Exceeded {timeout} seconds")
        return None
    except Exception as e:
        logger.error(f"Error fetching {url}: {str(e)}")
        return None


async def scrape_urls(urls: List[str], num_parallel: int = 10, limit: Optional[int] = None, show_progress: bool = False, timeout: Optional[float] = 5.0) -> AsyncIterator[Tuple[str, str, Optional[str]]]:
    """
    Asynchronously scrape a list of URLs with parallel processing, yielding results as they complete.

    Args:
        urls: List of URLs to scrape.
        num_parallel: Maximum number of concurrent requests (default: 10).
        limit: Optional limit on the number of URLs to process.
        show_progress: Whether to show a progress bar (default: False).
        timeout: Optional timeout in seconds for each HTTP request (default: 5.0).

    Yields:
        Tuple of (url, status, html_content), where status is "started" or "completed", and html_content is None for "started".
    """
    ua = UserAgent()
    semaphore = asyncio.Semaphore(num_parallel)
    urls_to_process = urls[:limit] if limit is not None else urls

    async def sem_fetch(url: str, session: aiohttp.ClientSession, pbar: Optional[tqdm_asyncio] = None) -> Tuple[str, Optional[str]]:
        async with semaphore:
            logger.debug(f"Starting to scrape URL: {url}")
            html = await scrape_url(session, url, ua, timeout)
            logger.debug(
                f"Completed scraping URL: {url}, success: {html is not None}")
            if pbar:
                pbar.update(1)
                active_tasks = min(num_parallel, len(
                    urls_to_process)) - semaphore._value
                pbar.set_description(f"Scraping URLs ({active_tasks} active)")
            return url, html

    async with aiohttp.ClientSession() as session:
        tasks = [sem_fetch(url, session) for url in urls_to_process]
        if show_progress:
            with tqdm_asyncio(total=len(urls_to_process), desc=f"Scraping URLs ({min(num_parallel, len(urls_to_process))} active)", file=sys.stdout, mininterval=0.1) as pbar:
                for task in asyncio.as_completed(tasks):
                    url, html = await task
                    yield url, "started", None
                    yield url, "completed", html
        else:
            for task in asyncio.as_completed(tasks):
                url, html = await task
                yield url, "started", None
                yield url, "completed", html


async def main():
    urls = [
        "https://www.imdb.com/list/ls505070747",
        "https://myanimelist.net/stacks/32507",
        "https://example.com",
        "https://python.org",
        "https://github.com",
        "https://httpbin.org/html",
        "https://www.wikipedia.org/",
        "https://www.mozilla.org",
        "https://www.stackoverflow.com"
    ]

    html_list = []
    async for url, status, html in scrape_urls(urls, num_parallel=5, limit=5, show_progress=True, timeout=5.0):
        if status == "completed":
            if not html:
                continue

            all_links = scrape_links(html, base_url=url)
            headers = get_md_header_contents(html)
            logger.success(
                f"Scraped {url}, headers length: {len(headers)}, links count: {len(all_links)}")

            html_list.append(html)

    logger.info(f"Scraped {len(html_list)} htmls")


if __name__ == "__main__":
    asyncio.run(main())
