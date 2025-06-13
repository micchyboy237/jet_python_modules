import requests
import aiohttp
import asyncio
from fake_useragent import UserAgent
from typing import AsyncIterator, List, Optional, Tuple
from jet.code.splitter_markdown_utils import get_md_header_contents
from jet.logger import logger
from jet.scrapers.utils import scrape_links
from jet.cache.redis.types import RedisConfigParams
from jet.cache.redis.utils import RedisCache

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


async def scrape_url(session: aiohttp.ClientSession, url: str, ua: UserAgent) -> Optional[str]:
    cache_key = f"html:{url}"
    cached_content = cache.get(cache_key)

    if cached_content:
        return cached_content['content']

    # logger.warning(f"Cache miss for {url}")

    try:
        headers = {'User-Agent': ua.random}
        async with session.get(url, headers=headers) as response:
            if response.status == 200:
                # logger.success(f"Scraped {url}")
                html_content = await response.text()
                cache.set(cache_key, {'content': html_content}, ttl=3600)
                return html_content
            else:
                logger.warning(
                    f"Failed: {url} - Status Code: {response.status}, Reason: {response.reason}")
                return None
    except Exception as e:
        logger.error(f"Error fetching {url}: {str(e)}")
        return None


async def scrape_urls(urls: List[str], num_parallel: int = 5) -> AsyncIterator[Tuple[str, str, Optional[str]]]:
    """
    Scrape URLs asynchronously and yield (url, status, html) for each URL.
    Status is 'started' when processing begins, 'completed' when done.
    HTML is None for 'started' status, and the scraped content or None for 'completed'.
    """
    ua = UserAgent()
    semaphore = asyncio.Semaphore(num_parallel)  # Keep this line

    async def sem_fetch(url: str, session: aiohttp.ClientSession) -> List[Tuple[str, str, Optional[str]]]:
        results = []
        async with semaphore:
            logger.debug(f"Starting to scrape URL: {url}")
            results.append((url, "started", None))
            html = await scrape_url(session, url, ua)
            logger.debug(
                f"Completed scraping URL: {url}, success: {html is not None}")
            results.append((url, "completed", html))
        return results

    async with aiohttp.ClientSession() as session:
        tasks = [sem_fetch(url, session) for url in urls]
        for task in asyncio.as_completed(tasks):
            task_results = await task
            for url, status, html in task_results:
                yield url, status, html

if __name__ == "__main__":
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
    results = asyncio.run(scrape_urls(urls, num_parallel=3))
    for url, html_str in zip(urls, results):
        if html_str:
            all_links = scrape_links(html_str, base_url=url)
            headers = get_md_header_contents(html_str)
            logger.success(
                f"Scraped {url}, headers length: {len(headers)}, links count: {len(all_links)}")
        else:
            logger.error(f"Failed to fetch {url}")
