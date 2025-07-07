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
from tqdm.asyncio import tqdm_asyncio

REDIS_CONFIG = RedisConfigParams(port=3102)
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


async def scrape_url(session: aiohttp.ClientSession, url: str, ua: UserAgent, timeout: Optional[float] = 5.0) -> Optional[str]:
    cache_key = f"html:{url}"
    cached_content = cache.get(cache_key)

    if cached_content:
        return cached_content['content']

    try:
        headers = {'User-Agent': ua.random}
        client_timeout = aiohttp.ClientTimeout(
            total=timeout) if timeout else None
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


async def scrape_urls(
    urls: List[str],
    num_parallel: int = 10,
    limit: Optional[int] = None,
    show_progress: bool = False,
    timeout: Optional[float] = 5.0
) -> AsyncIterator[Tuple[str, str, Optional[str]]]:
    ua = UserAgent()
    semaphore = asyncio.Semaphore(num_parallel)
    completed_count = 0

    async def sem_fetch_and_yield(url: str, session: aiohttp.ClientSession, pbar=None) -> List[Tuple[str, str, Optional[str]]]:
        results = []
        results.append((url, "started", None))
        async with semaphore:
            logger.debug(f"Starting to scrape URL: {url}")
            try:
                html = await scrape_url(session, url, ua, timeout)
                logger.debug(
                    f"Completed scraping URL: {url}, success: {html is not None}")

                if pbar:
                    pbar.update(1)
                    active_tasks = min(
                        num_parallel, len(urls)) - semaphore._value
                    pbar.set_description(
                        f"Scraping URLs ({active_tasks} active)")

                if html:
                    results.append((url, "completed", html))
                else:
                    results.append((url, "no_html", None))
            except Exception as e:
                logger.error(f"Exception while scraping {url}: {str(e)}")
                if pbar:
                    pbar.update(1)
                results.append((url, "error", None))
        return results

    async with aiohttp.ClientSession() as session:
        coroutines = [sem_fetch_and_yield(url, session) for url in urls]

        if show_progress:
            with tqdm_asyncio(total=len(urls), desc=f"Scraping URLs ({min(num_parallel, len(urls))} active)", file=sys.stdout, mininterval=0.1) as pbar:
                coroutines = [sem_fetch_and_yield(
                    url, session, pbar) for url in urls]

        tasks = [asyncio.create_task(coro) for coro in coroutines]

        for task in asyncio.as_completed(tasks):
            result = await task
            for item in result:
                yield item
                if item[1] == "completed":
                    completed_count += 1
                    if limit and completed_count >= limit:
                        logger.info(
                            f"Reached limit of {limit} completed URLs.")
                        for t in tasks:
                            if not t.done():
                                t.cancel()
                        return


async def consume_generator(gen: AsyncIterator[Tuple[str, str, Optional[str]]]) -> List[Tuple[str, str, Optional[str]]]:
    return [item async for item in gen]


async def main():
    urls = [
        "https://www.asfcxcvqawe.com",
        "https://www.imdb.com/list/ls505070747",
        "https://myanimelist.net/stacks/32507",
        "https://example.com",
        "https://python.org",
        "https://github.com",
        "https://httpbin.org/html",
        "https://www.wikipedia.org/",
        "https://www.mozilla.org",
        "https://www.stackoverflow.com",
    ]

    html_list = []
    async for url, status, html in scrape_urls(urls, num_parallel=3, limit=5, show_progress=True, timeout=5.0):
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
