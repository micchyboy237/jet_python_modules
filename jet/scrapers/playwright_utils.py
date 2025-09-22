import asyncio
import platform
import sys
import base64
from typing import AsyncIterator, List, Literal, Optional, TypedDict
from playwright.async_api import async_playwright, BrowserContext
from fake_useragent import UserAgent
from jet.cache.redis.types import RedisConfigParams
from jet.cache.redis.utils import RedisCache
from jet.logger import logger
from jet.scrapers.utils import scrape_links
from tqdm.asyncio import tqdm_asyncio

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
    timeout: Optional[float] = 5000,
    max_retries: int = 2,
    with_screenshot: bool = True,
    wait_for_js: bool = False,
    use_cache: bool = True
) -> ScrapeResult:
    cache_key = f"html:{url}"
    if use_cache:
        cached_content = cache.get(cache_key)
        if cached_content:
            logger.debug(f"Retrieved cached content for {url}")
            screenshot = cached_content.get('screenshot')
            if screenshot:
                try:
                    screenshot = base64.b64decode(screenshot)
                except Exception as e:
                    logger.error(f"Failed to decode cached screenshot for {url}: {str(e)}")
                    screenshot = None
                return {"url": url, "status": "completed", "html": cached_content['content'], "screenshot": screenshot}

    attempt = 0
    while attempt <= max_retries:
        try:
            page = await context.new_page()
            try:
                logger.debug(f"Navigating to {url}, attempt {attempt + 1}")
                await page.goto(url, timeout=timeout, wait_until="domcontentloaded")
                if wait_for_js:
                    logger.debug(f"Waiting for JS content on {url}")
                    await page.wait_for_timeout(5000)  # Fallback to 5-second wait
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
            finally:
                await page.close()
        except Exception as e:
            logger.error(f"Error fetching {url}: {str(e)} (Attempt {attempt + 1}/{max_retries + 1})")
            if attempt == max_retries:
                logger.debug(f"Max retries reached for {url}")
                return {"url": url, "status": "failed_no_html", "html": None, "screenshot": None}
            attempt += 1
            delay = 2 ** attempt
            logger.info(f"Retrying {url} after {delay} seconds")
            await asyncio.sleep(delay)
    return {"url": url, "status": "failed_no_html", "html": None, "screenshot": None}

async def scrape_urls(
    urls: List[str],
    num_parallel: int = 10,
    limit: Optional[int] = None,
    show_progress: bool = False,
    timeout: Optional[float] = 10000,  # Increased timeout
    max_retries: int = 2,
    with_screenshot: bool = True,
    headless: bool = True,
    wait_for_js: bool = False,
    use_cache: bool = True
) -> AsyncIterator[ScrapeResult]:
    semaphore = asyncio.Semaphore(num_parallel)
    completed_count = 0
    tasks = []
    async def sem_fetch_and_yield(url: str, context: BrowserContext, pbar=None) -> List[ScrapeResult]:
        results = []
        results.append({"url": url, "status": "started", "html": None, "screenshot": None})
        async with semaphore:
            try:
                result = await scrape_url(context, url, timeout, max_retries, with_screenshot, wait_for_js, use_cache)
                if pbar:
                    pbar.update(1)
                    active_tasks = min(num_parallel, len(urls)) - semaphore._value
                    pbar.set_description(f"Scraping URLs ({active_tasks} active)")
                results.append(result)
            except asyncio.CancelledError:
                logger.info(f"Task for {url} was cancelled")
                raise
            except Exception as e:
                logger.error(f"Exception while scraping {url}: {str(e)}")
                if pbar:
                    pbar.update(1)
                results.append({"url": url, "status": "failed_error", "html": None, "screenshot": None})
        return results

    async with async_playwright() as p:
        ua = UserAgent()
        browser = await p.chromium.launch(headless=headless)
        context = await browser.new_context(user_agent=ua.random)
        try:
            coroutines = [sem_fetch_and_yield(url, context, None) for url in urls]
            if show_progress:
                with tqdm_asyncio(total=len(urls), desc=f"Scraping URLs ({min(num_parallel, len(urls))} active)", file=sys.stdout, mininterval=0.1) as pbar:
                    coroutines = [sem_fetch_and_yield(url, context, pbar) for url in urls]
            tasks = [asyncio.create_task(coro) for coro in coroutines]
            try:
                for task in asyncio.as_completed(tasks):
                    try:
                        result = await task
                        for item in result:
                            yield item
                            if item["status"] == "completed":
                                completed_count += 1
                                if limit and completed_count >= limit:
                                    logger.info(f"Reached limit of {limit} completed URLs.")
                                    for t in tasks:
                                        if not t.done():
                                            t.cancel()
                                    await asyncio.gather(*tasks, return_exceptions=True)
                                    return
                    except asyncio.CancelledError:
                        logger.info("Task processing was cancelled")
                        raise
            except asyncio.CancelledError:
                logger.info("Cancelling all tasks due to interruption")
                for t in tasks:
                    if not t.done():
                        t.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
        finally:
            await context.close()
            await browser.close()
            # Ensure all tasks are cancelled and awaited
            for task in tasks:
                if not task.done():
                    task.cancel()
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
            except asyncio.CancelledError:
                logger.debug("All tasks cancelled and cleaned up")

async def consume_generator(gen: AsyncIterator[ScrapeResult]) -> List[ScrapeResult]:
    return [item async for item in gen]

def scrape_urls_sync(
    urls: List[str],
    num_parallel: int = 10,
    limit: Optional[int] = None,
    show_progress: bool = False,
    timeout: Optional[float] = 5000,
    max_retries: int = 2,
    with_screenshot: bool = True,
    headless: bool = True,
    wait_for_js: bool = False,
    use_cache: bool = True
) -> List[ScrapeResult]:
    async def run_scraper():
        return await consume_generator(
            scrape_urls(urls, num_parallel, limit, show_progress, timeout, max_retries, with_screenshot, headless, wait_for_js, use_cache)
        )

    if platform.system() == "Emscripten":
        return asyncio.ensure_future(run_scraper()).result()
    else:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(run_scraper())
        finally:
            if not loop.is_running():
                loop.close()
