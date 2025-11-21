import asyncio
import os
import sys
import base64
from typing import AsyncIterator, Iterator, List, Literal, Optional, TypedDict
from playwright.async_api import async_playwright, BrowserContext
from fake_useragent import UserAgent
from jet.cache.redis.types import RedisConfigParams
from jet.cache.redis.utils import RedisCache
from jet.logger import logger
from tqdm.asyncio import tqdm_asyncio

from jet.scrapers.browser.config import PLAYWRIGHT_CHROMIUM_EXECUTABLE
from jet.utils.inspect_utils import get_entry_file_dir

ScrapeStatus = Literal["started", "completed", "failed_no_html", "failed_error"]

class ScrapeResult(TypedDict):
    url: str
    status: ScrapeStatus
    html: Optional[str]
    screenshot: Optional[bytes]

REDIS_CONFIG = RedisConfigParams(port=6379)
cache = RedisCache(config=REDIS_CONFIG)

async def scrape_url(
    context: BrowserContext,
    url: str,
    timeout: Optional[float] = 10000,
    max_retries: int = 1,
    with_screenshot: bool = True,
    wait_for_js: bool = True,
    use_cache: bool = True  # Now respected properly
) -> ScrapeResult:
    cache_key = f"html:{url}"
    if use_cache:
        cached_content = cache.get(cache_key)
        if cached_content:
            logger.debug(f"Retrieved cached content for {url}")
            screenshot = None
            if with_screenshot and 'screenshot' in cached_content:
                try:
                    screenshot = base64.b64decode(cached_content['screenshot'])
                except Exception as e:
                    logger.error(f"Failed to decode cached screenshot for {url}: {str(e)}")
            return {"url": url, "status": "completed", "html": cached_content['content'], "screenshot": screenshot}

    attempt = 0
    page = None
    try:
        while attempt <= max_retries:
            try:
                page = await context.new_page()
                logger.debug(f"Navigating to {url}, attempt {attempt + 1}")
                await page.goto(url, timeout=timeout, wait_until="domcontentloaded")
                if wait_for_js:
                    logger.debug(f"Waiting for JS content on {url}")
                    await page.wait_for_timeout(5000)
                html_content = await page.content()
                screenshot = None
                if with_screenshot:
                    screenshot = await page.screenshot(full_page=True)
                if use_cache:
                    cache_data = {'content': html_content}
                    if screenshot:
                        cache_data['screenshot'] = base64.b64encode(screenshot).decode('utf-8')
                        logger.debug(f"Encoded screenshot for {url}, length: {len(cache_data['screenshot'])}")
                    cache.set(cache_key, cache_data, ttl=3600)
                    logger.debug(f"Cached content for {url}")
                return {"url": url, "status": "completed", "html": html_content, "screenshot": screenshot}
            except Exception as e:
                logger.error(f"Error fetching {url}: {str(e)} (Attempt {attempt + 1}/{max_retries + 1})")
                if attempt == max_retries:
                    logger.debug(f"Max retries reached for {url}")
                    return {"url": url, "status": "failed_no_html", "html": None, "screenshot": None}
                attempt += 1
                try:
                    # Use a cancellation-safe delay
                    await asyncio.wait_for(asyncio.sleep(2 ** attempt), timeout=10.0)
                except asyncio.CancelledError:
                    logger.info(f"Retry delay for {url} cancelled")
                    raise
                except asyncio.TimeoutError:
                    logger.warning(f"Retry delay timeout for {url}")
                    return {"url": url, "status": "failed_no_html", "html": None, "screenshot": None}
            finally:
                if page:
                    try:
                        await asyncio.wait_for(page.close(), timeout=5.0)
                    except (asyncio.CancelledError, asyncio.TimeoutError, RuntimeError) as e:
                        logger.debug(f"Failed to close page for {url}: {str(e)}")
    except asyncio.CancelledError:
        logger.info(f"Scrape task for {url} cancelled")
        if page:
            try:
                await asyncio.wait_for(page.close(), timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError, RuntimeError) as e:
                logger.debug(f"Failed to close page for {url} during cancellation: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in scrape_url for {url}: {str(e)}")
        return {"url": url, "status": "failed_error", "html": None, "screenshot": None}

async def scrape_urls(
    urls: List[str],
    num_parallel: int = 10,
    limit: Optional[int] = None,
    show_progress: bool = False,
    timeout: Optional[float] = 10000,
    max_retries: int = 1,
    with_screenshot: bool = True,
    headless: bool = True,
    wait_for_js: bool = True,
    use_cache: bool = True
) -> AsyncIterator[ScrapeResult]:
    semaphore = asyncio.Semaphore(num_parallel)
    completed_count = 0
    tasks = []

    # Helper to yield cached result immediately
    def yield_cached(url: str, cached_data: dict) -> ScrapeResult:
        screenshot = None
        if with_screenshot and 'screenshot' in cached_data:
            try:
                screenshot = base64.b64decode(cached_data['screenshot'])
            except Exception as e:
                logger.error(f"Failed to decode cached screenshot for {url}: {e}")
        logger.debug(f"Cache hit for {url}")
        return {"url": url, "status": "completed", "html": cached_data['content'], "screenshot": screenshot}

    async def sem_fetch_and_yield(url: str, context: BrowserContext, pbar=None) -> List[ScrapeResult]:
        results = []
        results.append({"url": url, "status": "started", "html": None, "screenshot": None})
        async with semaphore:
            try:
                result = await scrape_url(
                    context, url, timeout, max_retries,
                    with_screenshot, wait_for_js, use_cache=False  # Important: disable inner cache check
                )
                if pbar:
                    pbar.update(1)
                results.append(result)
            except Exception as e:
                logger.error(f"Exception while scraping {url}: {str(e)}")
                if pbar:
                    pbar.update(1)
                results.append({"url": url, "status": "failed_error", "html": None, "screenshot": None})
        return results

    async with async_playwright() as p:
        ua = UserAgent()
        browser = None
        context = None
        try:
            traces_dir = f"{get_entry_file_dir()}/playwright/traces"
            os.makedirs(traces_dir, exist_ok=True)
            browser = await p.chromium.launch(
                headless=headless,
                executable_path=PLAYWRIGHT_CHROMIUM_EXECUTABLE,
                traces_dir=traces_dir,
            )
            context = await browser.new_context(user_agent=ua.random)

            # Pre-check cache for all URLs
            urls_to_scrape = []
            for url in urls:
                if use_cache:
                    cache_key = f"html:{url}"
                    cached = cache.get(cache_key)
                    if cached:
                        yield yield_cached(url, cached)
                        if show_progress:
                            # Still update progress for cached items
                            pass  # will be handled below via initial pbar total
                        completed_count += 1
                        if limit and completed_count >= limit:
                            return
                        continue
                urls_to_scrape.append(url)

            # Only launch browser + tasks for URLs that need scraping
            if not urls_to_scrape:
                return

            desc = f"Scraping URLs ({num_parallel} max active)"
            if show_progress:
                with tqdm_asyncio(total=len(urls_to_scrape), desc=desc, file=sys.stdout, mininterval=0.1) as pbar:
                    tasks = [asyncio.create_task(sem_fetch_and_yield(url, context, pbar)) for url in urls_to_scrape]
                    for task in asyncio.as_completed(tasks):
                        result_list = await task
                        for item in result_list:
                            if item["status"] == "started":
                                continue
                            yield item
                            if item["status"] == "completed":
                                completed_count += 1
                                if limit and completed_count >= limit:
                                    for t in tasks:
                                        if not t.done():
                                            t.cancel()
                                    await asyncio.gather(*tasks, return_exceptions=True)
                                    return
            else:
                tasks = [asyncio.create_task(sem_fetch_and_yield(url, context, None)) for url in urls_to_scrape]
                for task in asyncio.as_completed(tasks):
                    result_list = await task
                    for item in result_list:
                        if item["status"] == "started":
                            continue
                        yield item
                        if item["status"] == "completed":
                            completed_count += 1
                            if limit and completed_count >= limit:
                                return

            await asyncio.gather(*tasks, return_exceptions=True)

        finally:
            if context:
                await asyncio.wait_for(context.close(), timeout=5.0)
            if browser:
                await asyncio.wait_for(browser.close(), timeout=5.0)

def scrape_urls_sync(
    urls: List[str],
    num_parallel: int = 10,
    limit: Optional[int] = None,
    show_progress: bool = False,
    timeout: Optional[float] = 10000,
    max_retries: int = 1,
    with_screenshot: bool = True,
    headless: bool = True,
    wait_for_js: bool = True,
    use_cache: bool = True
) -> Iterator[ScrapeResult]:
    """
    Synchronously scrape a list of URLs using Playwright, yielding ScrapeResult for each URL.
    
    Args:
        urls: List of URLs to scrape.
        num_parallel: Number of parallel browser tasks.
        limit: Maximum number of successful scrapes to return (None for no limit).
        show_progress: Whether to show a progress bar.
        timeout: Timeout for page navigation in milliseconds.
        max_retries: Number of retries for failed attempts.
        with_screenshot: Whether to capture screenshots.
        headless: Whether to run browser in headless mode.
        wait_for_js: Whether to wait for JS rendering.
        use_cache: Whether to use Redis caching.
    
    Yields:
        ScrapeResult: Dictionary containing URL, status, HTML content, and optional screenshot.
    """
    async def run_scrape() -> List[ScrapeResult]:
        results = []
        async for result in scrape_urls(
            urls=urls,
            num_parallel=num_parallel,
            limit=limit,
            show_progress=show_progress,
            timeout=timeout,
            max_retries=max_retries,
            with_screenshot=with_screenshot,
            headless=headless,
            wait_for_js=wait_for_js,
            use_cache=use_cache
        ):
            results.append(result)
        return results

    loop = asyncio.get_event_loop()
    if loop.is_running():
        logger.warning("Cannot run synchronous scraping in an already running event loop.")
        raise RuntimeError("Synchronous scraping requires a non-running event loop.")
    
    try:
        results = loop.run_until_complete(run_scrape())
        for result in results:
            yield result
    except Exception as e:
        logger.error(f"Error in synchronous URL scraping: {str(e)}")
        raise
