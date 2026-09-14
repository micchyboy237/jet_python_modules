import asyncio
import os
import shutil
from typing import List, Optional, TypedDict

from fake_useragent import UserAgent
from jet.cache.redis.types import RedisConfigParams
from jet.cache.redis.utils import RedisCache
from jet.logger import logger
from jet.scrapers.browser.config import (
    PLAYWRIGHT_CHROMIUM_EXECUTABLE,
)
from jet.utils.inspect_utils import get_entry_file_dir
from playwright.async_api import Browser as AsyncBrowser
from playwright.async_api import Page as AsyncPage
from playwright.async_api import async_playwright
from playwright.sync_api import Browser as SyncBrowser
from playwright.sync_api import Page as SyncPage
from playwright.sync_api import sync_playwright
from playwright_stealth import Stealth
from tqdm.asyncio import tqdm

REDIS_CONFIG = RedisConfigParams(port=6379)
browser_page = None


class PageDimensions(TypedDict):
    width: int
    height: int
    deviceScaleFactor: float


class PageContent(TypedDict):
    url: str
    dimensions: PageDimensions
    screenshot: str
    html: str


def setup_browser_page(page: Optional[SyncPage] = None, headless: bool = True):
    global browser_page
    if not browser_page:
        browser_page = page or setup_sync_browser_page(headless=headless)
    return browser_page


async def asetup_browser_page(*, headless: bool = True) -> AsyncPage:
    """Sets up an asynchronous Playwright browser page with anti-detection settings."""
    browser = await setup_async_browser_session(headless=headless)
    return await browser.new_page()


def setup_sync_browser_session(*, headless: bool = True) -> SyncBrowser:
    """Sets up a synchronous Playwright browser session with v2.x stealth applied to persistent context."""
    logger.log("Initializing sync Playwright with stealth...", colors=["BLUE", "INFO"])

    playwright = sync_playwright().start()
    ua = UserAgent()
    generated_dir = os.path.join(get_entry_file_dir(), "generated")
    user_data_dir = os.path.join(generated_dir, "browser_context")
    shutil.rmtree(user_data_dir, ignore_errors=True)

    logger.log(
        f"Launching persistent context at: {user_data_dir}", colors=["GRAY", "DEBUG"]
    )
    context = playwright.chromium.launch_persistent_context(
        user_data_dir=user_data_dir,
        headless=headless,
        executable_path=PLAYWRIGHT_CHROMIUM_EXECUTABLE,
        user_agent=ua.random,
        viewport={"width": 1440, "height": 900, "deviceScaleFactor": 0.9},
    )

    # Manually apply stealth to the persistent context
    stealth = Stealth()
    try:
        stealth.apply_stealth_sync(context)
        logger.log(
            "Stealth evasions applied successfully to persistent context",
            colors=["GREEN", "SUCCESS"],
        )
    except Exception as e:
        logger.log(
            f"Failed to apply stealth to persistent context: {e}",
            colors=["RED", "ERROR"],
        )

    return context


async def setup_async_browser_session(*, headless: bool = True) -> AsyncBrowser:
    """Sets up an asynchronous Playwright browser session with v2.x stealth applied to persistent context."""
    logger.log("Initializing async Playwright with stealth...", colors=["BLUE", "INFO"])

    # NOTE: Stealth().use_async() does NOT support launch_persistent_context yet [[1]].
    # We must apply stealth manually to the persistent context.
    playwright = await async_playwright().start()
    ua = UserAgent()
    generated_dir = os.path.join(get_entry_file_dir(), "generated")
    user_data_dir = os.path.join(generated_dir, "browser_context")
    shutil.rmtree(user_data_dir, ignore_errors=True)

    logger.log(
        f"Launching persistent context at: {user_data_dir}", colors=["GRAY", "DEBUG"]
    )
    context = await playwright.chromium.launch_persistent_context(
        user_data_dir=user_data_dir,
        headless=headless,
        executable_path=PLAYWRIGHT_CHROMIUM_EXECUTABLE,
        user_agent=ua.random,
        viewport={"width": 1440, "height": 900, "deviceScaleFactor": 0.9},
    )

    # Manually apply stealth to the persistent context
    stealth = Stealth()
    try:
        await stealth.apply_stealth_async(context)
        logger.log(
            "Stealth evasions applied successfully to persistent context",
            colors=["GREEN", "SUCCESS"],
        )
    except Exception as e:
        logger.log(
            f"Failed to apply stealth to persistent context: {e}",
            colors=["RED", "ERROR"],
        )
        # Continue execution but warn that stealth may be incomplete

    return context


def setup_sync_browser_page(*, headless: bool = False) -> SyncPage:
    """Sets up a synchronous Playwright browser session and returns the browser instance."""
    browser = setup_sync_browser_session(headless=headless)
    return browser.new_page()


async def setup_async_browser_page(*, headless: bool = False) -> AsyncPage:
    """Sets up an asynchronous Playwright browser page and returns the browser instance."""
    browser = await setup_async_browser_session(headless=headless)
    return await browser.new_page()


def fetch_page_content_sync(
    url: str,
    wait_for_css: Optional[List[str]],
    max_wait_timeout: int = 10000,
    headless: bool = True,
    use_cache: bool = False,
) -> PageContent:
    """Fetches page content synchronously, including screenshot and HTML."""
    cache = RedisCache(config=REDIS_CONFIG)
    cache_key = url
    cached_result = cache.get(cache_key) if use_cache else None
    browser_page = setup_browser_page(headless=headless)
    if cached_result:
        logger.log(
            "scrape_url: Cache hit for", cache_key, colors=["LOG", "BRIGHT_SUCCESS"]
        )
        return cached_result
    if wait_for_css:
        logger.log("Waiting for elements css:", wait_for_css, colors=["GRAY", "DEBUG"])
        for css_selector in wait_for_css:
            browser_page.wait_for_selector(css_selector, timeout=max_wait_timeout)
    generated_dir = os.path.join(get_entry_file_dir(), "generated")
    screenshot_path = f"{generated_dir}/example.png"
    browser_page.screenshot(path=screenshot_path)
    dimensions: PageDimensions = browser_page.evaluate("""() => ({
        width: document.documentElement.clientWidth,
        height: document.documentElement.clientHeight,
        deviceScaleFactor: window.devicePixelRatio
    })""")
    result: PageContent = {
        "url": url,
        "dimensions": dimensions,
        "screenshot": os.path.realpath(screenshot_path),
        "html": browser_page.content(),
    }
    if use_cache:
        cache.set(cache_key, result)
    return result


async def fetch_page_content_async(
    url: str,
    wait_for_css: Optional[List[str]],
    page: Optional[AsyncPage] = None,
    max_wait_timeout: int = 10000,
    headless: bool = True,
    use_cache: bool = False,
) -> PageContent:
    """Fetches page content asynchronously, including screenshot and HTML."""
    cache = RedisCache(config=REDIS_CONFIG)
    cache_key = url
    cached_result = cache.get(cache_key) if use_cache else None
    browser_page = page or await asetup_browser_page(headless=headless)
    try:
        if cached_result:
            logger.log(
                "scrape_url: Cache hit for", cache_key, colors=["LOG", "BRIGHT_SUCCESS"]
            )
            return cached_result
        if wait_for_css:
            logger.log(
                "Waiting for elements css:", wait_for_css, colors=["GRAY", "DEBUG"]
            )
            for css_selector in wait_for_css:
                await browser_page.wait_for_selector(
                    css_selector, timeout=max_wait_timeout
                )
        generated_dir = os.path.join(get_entry_file_dir(), "generated")
        screenshot_path = f"{generated_dir}/example.png"
        await browser_page.screenshot(path=screenshot_path)
        dimensions: PageDimensions = await browser_page.evaluate("""() => ({
            width: document.documentElement.clientWidth,
            height: document.documentElement.clientHeight,
            deviceScaleFactor: window.devicePixelRatio
        })""")
        result: PageContent = {
            "url": url,
            "dimensions": dimensions,
            "screenshot": os.path.realpath(screenshot_path),
            "html": await browser_page.content(),
        }
        if use_cache:
            cache.set(cache_key, result)
        return result
    finally:
        if not page:
            await browser_page.close()


def scrape_sync(
    url: str,
    wait_for_css: Optional[List[str]] = None,
    headless: bool = True,
    use_cache: bool = False,
) -> PageContent:
    """Scrapes a webpage synchronously."""
    browser_page = setup_browser_page(headless=headless)
    browser_page.goto(url, wait_until="domcontentloaded")
    return fetch_page_content_sync(url, wait_for_css, use_cache=use_cache)


async def scrape_async(
    url: str,
    wait_for_css: Optional[List[str]] = None,
    headless: bool = True,
    use_cache: bool = False,
) -> PageContent:
    """Scrapes a webpage asynchronously."""
    browser_page = await asetup_browser_page(headless=headless)
    try:
        await browser_page.goto(url)
        return await fetch_page_content_async(
            url, wait_for_css, page=browser_page, use_cache=use_cache
        )
    finally:
        await browser_page.close()


async def setup_browser_pool(
    max_pages: int = 2, headless: bool = False
) -> List[AsyncPage]:
    """Creates a pool of browser pages to be shared among tasks."""
    browser = await setup_async_browser_session(headless=headless)
    return [await browser.new_page() for _ in range(max_pages)]


async def scrape_async_limited(
    urls: List[str], max_concurrent_tasks: int = 2, headless: bool = False
) -> List[PageContent]:
    """Scrapes multiple URLs asynchronously, limiting concurrent tasks while sharing browser pages."""
    pages = await setup_browser_pool(max_concurrent_tasks, headless)
    page_queue = asyncio.Queue()
    for page in pages:
        await page_queue.put(page)
    results = []
    progress_bar = tqdm(total=len(urls), desc="Scraping Progress", unit="url")

    async def bound_scrape(url) -> PageContent:
        """Scrape a single URL using an available browser page from the queue."""
        page = await page_queue.get()
        try:
            result = await scrape_async(url)
            results.append(result)
            progress_bar.update(1)
        finally:
            await page_queue.put(page)
        return result

    semaphore = asyncio.Semaphore(max_concurrent_tasks)

    async def controlled_scrape(url):
        async with semaphore:
            return await bound_scrape(url)

    await asyncio.gather(*(controlled_scrape(url) for url in urls))
    for page in pages:
        await page.close()
    progress_bar.close()
    return results


if __name__ == "__main__":
    urls_to_scrape = [
        "https://example.com",
        "https://example.org",
        "https://example.net",
        "https://example.info",
    ]
    asyncio.run(
        scrape_async_limited(
            urls=urls_to_scrape,
            max_concurrent_tasks=2,
            headless=True,
        )
    )
