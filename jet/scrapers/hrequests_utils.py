import asyncio
import platform
import sys
from collections.abc import AsyncIterator
from typing import Literal

import aiohttp
import requests
from fake_useragent import UserAgent
from jet.cache.redis.types import RedisConfigParams
from jet.cache.redis.utils import RedisCache
from jet.logger import logger
from jet.scrapers.utils import scrape_links
from tqdm.asyncio import tqdm_asyncio

ScrapeStatus = Literal["started", "completed", "failed_no_html", "failed_error"]

REDIS_CONFIG = RedisConfigParams(port=6379)
cache = RedisCache(config=REDIS_CONFIG)


def scrape_url_sync(url: str, timeout: float | None = 15.0) -> str | None:
    cache_key = f"html:{url}"
    cached_content = cache.get(cache_key)

    if cached_content:
        return cached_content["content"]

    try:
        ua = UserAgent()
        headers = {"User-Agent": ua.random}
        response = requests.get(url, headers=headers, timeout=timeout)
        if response.status_code == 200:
            html_content = response.text
            cache.set(cache_key, {"content": html_content}, ttl=3600)
            return html_content
        else:
            logger.warning(
                f"Failed: {url} - Status Code: {response.status_code}, Reason: {response.reason}"
            )
            return None
    except Exception as e:
        logger.error(f"Error fetching {url}: {str(e)}")
        return None


async def scrape_url(
    session: aiohttp.ClientSession,
    url: str,
    ua: UserAgent,
    timeout: float | None = 15.0,  # ✅ Increased from 5.0
    max_retries: int = 2,  # ✅ Increased from 1
) -> str | None:
    cache_key = f"html:{url}"
    cached_content = cache.get(cache_key)

    if cached_content:
        return cached_content["content"]

    attempt = 0
    while attempt <= max_retries:
        try:
            headers = {"User-Agent": ua.random}
            # ✅ Use separate connect/read timeouts instead of just total
            client_timeout = (
                aiohttp.ClientTimeout(
                    total=timeout,
                    connect=10.0,
                    sock_read=timeout,
                )
                if timeout
                else None
            )

            async with session.get(
                url, headers=headers, timeout=client_timeout, ssl=False
            ) as response:
                if response.status == 200:
                    html_content = await response.text()
                    cache.set(cache_key, {"content": html_content}, ttl=3600)
                    return html_content
                elif response.status in (403, 429, 503):
                    # ✅ Don't retry on permanent blocks
                    logger.warning(f"Blocked: {url} - Status {response.status}")
                    return None
                else:
                    logger.warning(
                        f"Failed: {url} - Status {response.status}, Reason: {response.reason}"
                    )
                    return None

        except asyncio.TimeoutError:
            logger.warning(  # ✅ Changed from error to warning for retries
                f"Timeout fetching {url}: Exceeded {timeout}s (Attempt {attempt + 1}/{max_retries + 1})"
            )
            if attempt == max_retries:
                logger.error(f"All retries exhausted for {url}")
                return None
        except Exception as e:
            logger.warning(
                f"Error fetching {url}: {str(e)} (Attempt {attempt + 1}/{max_retries + 1})"
            )
            if attempt == max_retries:
                logger.error(f"All retries exhausted for {url}")
                return None

        attempt += 1
        delay = min(2**attempt, 8)  # ✅ Cap backoff at 8 seconds
        logger.info(f"Retrying {url} after {delay}s")
        await asyncio.sleep(delay)

    return None


async def scrape_urls(
    urls: list[str],
    num_parallel: int = 10,
    limit: int | None = None,
    show_progress: bool = False,
    timeout: float | None = 15.0,  # ✅ Match new default
    max_retries: int = 2,  # ✅ Match new default
) -> AsyncIterator[tuple[str, ScrapeStatus, str | None]]:
    ua = UserAgent()
    semaphore = asyncio.Semaphore(num_parallel)
    completed_count = 0
    tasks: list[asyncio.Task] = []
    limit_reached = False  # ✅ Track intentional cancellation

    async def sem_fetch_and_yield(url, session, pbar=None):
        results = [(url, "started", None)]
        async with semaphore:
            try:
                html = await scrape_url(session, url, ua, timeout, max_retries)
                if pbar:
                    pbar.update(1)
                    active = min(num_parallel, len(urls)) - semaphore._value
                    pbar.set_description(f"Scraping URLs ({active} active)")
                status = "completed" if html else "failed_no_html"
                results.append((url, status, html))
            except asyncio.CancelledError:
                raise  # ✅ Re-raise silently; handled by caller
            except Exception as e:
                logger.error(f"Exception scraping {url}: {e}")
                if pbar:
                    pbar.update(1)
                results.append((url, "failed_error", None))
        return results

    async def _cleanup():
        """Cancel remaining tasks without noisy logs."""
        for t in tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async with aiohttp.ClientSession() as session:
        try:
            pbar_ctx = (
                tqdm_asyncio(
                    total=len(urls),
                    desc=f"Scraping URLs ({min(num_parallel, len(urls))} active)",
                    file=sys.stdout,
                    mininterval=0.1,
                )
                if show_progress
                else None
            )

            with pbar_ctx as pbar:
                coroutines = [sem_fetch_and_yield(url, session, pbar) for url in urls]
                tasks = [asyncio.create_task(c) for c in coroutines]

                for task in asyncio.as_completed(tasks):
                    try:
                        result_list = await task
                        for item in result_list:
                            yield item
                            if item[1] == "completed":
                                completed_count += 1
                                if limit and completed_count >= limit:
                                    logger.info(
                                        f"Reached limit of {limit} completed URLs."
                                    )
                                    limit_reached = True  # ✅ Mark as intentional
                                    await _cleanup()
                                    return
                    except asyncio.CancelledError:
                        if not limit_reached:
                            logger.debug("Task cancelled during iteration")
                        raise

        except asyncio.CancelledError:
            if not limit_reached:
                logger.info("scrape_urls externally cancelled, cleaning up")
            await _cleanup()
            raise
        finally:
            # ✅ Only cleanup if we haven't already via limit/cancellation
            if not limit_reached:
                await _cleanup()


def scrape_urls_sync(
    urls: list[str],
    num_parallel: int = 10,
    limit: int | None = None,
    show_progress: bool = False,
    timeout: float | None = 15.0,
    max_retries: int = 2,
) -> list[tuple[str, ScrapeStatus, str | None]]:
    """
    Synchronously scrape multiple URLs while leveraging async parallel capabilities.
    Handles event loop creation for both standard Python and Pyodide (Emscripten).

    Args:
        urls: List of URLs to scrape
        num_parallel: Number of parallel requests
        limit: Maximum number of successful responses to collect
        show_progress: Whether to show a progress bar
        timeout: Timeout per request in seconds
        max_retries: Maximum number of retries per URL

    Returns:
        List of tuples containing (url, status, html_content)
    """

    async def run_scraper():
        return await consume_generator(
            scrape_urls(urls, num_parallel, limit, show_progress, timeout, max_retries)
        )

    if platform.system() == "Emscripten":
        # Pyodide environment: Use ensure_future for async execution
        return asyncio.ensure_future(run_scraper()).result()
    else:
        # Standard Python: Get or create event loop
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            return loop.run_until_complete(run_scraper())
        finally:
            # Clean up event loop if we created it
            if not loop.is_running():
                loop.close()


async def consume_generator(
    gen: AsyncIterator[tuple[str, ScrapeStatus, str | None]],
) -> list[tuple[str, ScrapeStatus, str | None]]:
    return [item async for item in gen]


def main(urls):
    """
    Synchronously scrape the given URLs and log summary information.
    """
    # Use the synchronous scrape_urls_sync function to get results
    results = scrape_urls_sync(
        urls, num_parallel=3, limit=5, show_progress=True, timeout=5.0, max_retries=3
    )
    html_list = []
    for url, status, html in results:
        if status == "completed":
            if not html:
                continue
            all_links = scrape_links(html, base_url=url)
            logger.success(f"Scraped {url}, links count: {len(all_links)}")
            html_list.append(html)
    logger.info(f"Done sync scraped {len(html_list)} htmls")


async def amain(urls):
    html_list = []
    async for url, status, html in scrape_urls(
        urls, num_parallel=3, limit=5, show_progress=True, timeout=5.0, max_retries=3
    ):
        if status == "completed":
            if not html:
                continue
            all_links = scrape_links(html, base_url=url)
            logger.success(f"Scraped {url}, links count: {len(all_links)}")
            html_list.append(html)

    logger.info(f"Done async scraped {len(html_list)} htmls")


if __name__ == "__main__":
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

    main(urls)
    asyncio.run(amain(urls))
