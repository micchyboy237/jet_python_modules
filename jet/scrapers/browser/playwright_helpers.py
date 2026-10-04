import asyncio
import base64
import os
import shutil
import sys
from typing import AsyncIterator, Iterator, List, Literal, Optional, TypedDict

from fake_useragent import UserAgent
from jet.cache.redis.types import RedisConfigParams
from jet.cache.redis.utils import RedisCache
from jet.logger import logger
from jet.scrapers.browser.config import (
    PLAYWRIGHT_CHROMIUM_EXECUTABLE,
)
from jet.utils.inspect_utils import get_entry_file_dir
from playwright.async_api import Browser as AsyncBrowser
from playwright.async_api import BrowserContext, async_playwright
from playwright.async_api import Page as AsyncPage
from playwright.sync_api import Browser as SyncBrowser
from playwright.sync_api import Page as SyncPage
from playwright.sync_api import sync_playwright
from playwright_stealth import Stealth
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

REDIS_CONFIG = RedisConfigParams(port=6379)
browser_page = None

ScrollStrategy = Literal["none", "once", "until_stable"]
ScrollMode = Literal["jump", "increment"]
ScrapeStatus = Literal["started", "completed", "failed_no_html", "failed_error"]


class PageDimensions(TypedDict):
    width: int
    height: int
    deviceScaleFactor: float


class PageContent(TypedDict):
    url: str
    dimensions: PageDimensions
    screenshot: str
    html: str


class ScrapeResult(TypedDict):
    url: str
    status: ScrapeStatus
    html: Optional[str]
    screenshot: Optional[bytes]


cache = RedisCache(config=REDIS_CONFIG)


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


# ============================================================================
# NEW PARALLEL SCRAPING FUNCTIONS (similar to playwright_utils.py)
# ============================================================================


async def scroll_to_bottom(
    page,
    *,
    strategy: ScrollStrategy = "until_stable",
    mode: ScrollMode = "increment",
    max_attempts: int = 15,
    delay_ms: int = 1400,
    overall_timeout_ms: int = 45000,
) -> None:
    """Scroll to bottom of page with various strategies."""
    if strategy == "none":
        return

    logger.debug(f"Starting scroll_to_bottom ({strategy}, mode={mode})")
    start_time = asyncio.get_event_loop().time()
    last_height = await page.evaluate("document.documentElement.scrollHeight")
    STABILIZE_THRESHOLD_PX = 150
    no_change_count = 0
    MAX_NO_CHANGE = 4

    for attempt in range(1, max_attempts + 1):
        if (
            overall_timeout_ms
            and (asyncio.get_event_loop().time() - start_time) * 1000
            > overall_timeout_ms
        ):
            logger.warning("Scroll timeout reached")
            break

        if mode == "jump":
            await page.evaluate(
                "window.scrollTo(0, document.documentElement.scrollHeight)"
            )
        else:
            await page.evaluate("window.scrollBy(0, window.innerHeight * 0.9)")

        await page.wait_for_timeout(delay_ms)

        if strategy == "once":
            break

        metrics = await page.evaluate("""() => ({
            scrollHeight: document.documentElement.scrollHeight,
            scrollY: window.scrollY,
            innerHeight: window.innerHeight
        })""")

        new_height = metrics["scrollHeight"]
        view_bottom = metrics["scrollY"] + metrics["innerHeight"]
        bottom_reached = view_bottom >= (new_height - STABILIZE_THRESHOLD_PX)
        height_changed = new_height != last_height

        logger.debug(
            f"Attempt {attempt}/{max_attempts} | "
            f"scrollY={metrics['scrollY']}, viewBottom={view_bottom}, "
            f"scrollHeight={new_height}, bottom_reached={bottom_reached}, "
            f"height_changed={height_changed}"
        )

        if not height_changed:
            no_change_count += 1
        else:
            no_change_count = 0

        if mode == "increment":
            if bottom_reached:
                if height_changed:
                    logger.debug("Near bottom but height still changing → continue")
                else:
                    logger.debug(
                        f"Reached near bottom and height stable after {attempt} attempts"
                    )
                    break
            elif no_change_count >= MAX_NO_CHANGE:
                logger.warning(
                    f"Height not changing for {no_change_count} attempts, but not at bottom yet → stopping to avoid infinite loop"
                )
                break
        else:
            if not height_changed:
                logger.debug(
                    f"Scroll stabilized (height only) after {attempt} attempts"
                )
                break

        last_height = new_height
    else:
        logger.info(
            f"Reached max scroll attempts ({max_attempts}) without stabilization"
        )


async def _scrape_single_url(
    context: BrowserContext,
    url: str,
    *,
    scroll_strategy: ScrollStrategy = "until_stable",
    scroll_max_attempts: int = 15,
    scroll_delay_ms: int = 1400,
    scroll_timeout_ms: int = 45000,
    timeout: Optional[float] = 10000,
    max_retries: int = 1,
    with_screenshot: bool = True,
    scroll_mode: ScrollMode = "increment",
    wait_for_js: bool = True,
    use_cache: bool = False,
) -> ScrapeResult:
    """Scrape a single URL with full feature set."""
    cache_key = f"html:{url}"

    if use_cache:
        cached_content = cache.get(cache_key)
        if cached_content:
            logger.debug(f"Retrieved cached content for {url}")
            screenshot = None
            if with_screenshot and "screenshot" in cached_content:
                try:
                    screenshot = base64.b64decode(cached_content["screenshot"])
                except Exception as e:
                    logger.error(
                        f"Failed to decode cached screenshot for {url}: {str(e)}"
                    )
            return {
                "url": url,
                "status": "completed",
                "html": cached_content["content"],
                "screenshot": screenshot,
            }

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
                    await page.wait_for_timeout(3500)

                if scroll_strategy != "none":
                    await scroll_to_bottom(
                        page,
                        strategy=scroll_strategy,
                        max_attempts=scroll_max_attempts,
                        delay_ms=scroll_delay_ms,
                        mode=scroll_mode,
                        overall_timeout_ms=scroll_timeout_ms,
                    )

                html_content = await page.content()
                screenshot = None
                if with_screenshot:
                    screenshot = await page.screenshot(full_page=True)

                if use_cache:
                    cache_data = {"content": html_content}
                    if screenshot:
                        cache_data["screenshot"] = base64.b64encode(screenshot).decode(
                            "utf-8"
                        )
                        logger.debug(
                            f"Encoded screenshot for {url}, length: {len(cache_data['screenshot'])}"
                        )
                    cache.set(cache_key, cache_data, ttl=3600)
                    logger.debug(f"Cached content for {url}")

                return {
                    "url": url,
                    "status": "completed",
                    "html": html_content,
                    "screenshot": screenshot,
                }

            except Exception as e:
                logger.error(
                    f"Error fetching {url}: {str(e)} (Attempt {attempt + 1}/{max_retries + 1})"
                )
                if attempt == max_retries:
                    logger.debug(f"Max retries reached for {url}")
                    return {
                        "url": url,
                        "status": "failed_no_html",
                        "html": None,
                        "screenshot": None,
                    }
                attempt += 1
                try:
                    await asyncio.wait_for(asyncio.sleep(2**attempt), timeout=10.0)
                except asyncio.CancelledError:
                    logger.info(f"Retry delay for {url} cancelled")
                    raise
                except asyncio.TimeoutError:
                    logger.warning(f"Retry delay timeout for {url}")
                    return {
                        "url": url,
                        "status": "failed_no_html",
                        "html": None,
                        "screenshot": None,
                    }

            finally:
                if page:
                    try:
                        await asyncio.wait_for(page.close(), timeout=5.0)
                    except (
                        asyncio.CancelledError,
                        asyncio.TimeoutError,
                        RuntimeError,
                    ) as e:
                        logger.debug(f"Failed to close page for {url}: {str(e)}")

    except asyncio.CancelledError:
        logger.info(f"Scrape task for {url} cancelled")
        if page:
            try:
                await asyncio.wait_for(page.close(), timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError, RuntimeError) as e:
                logger.debug(
                    f"Failed to close page for {url} during cancellation: {str(e)}"
                )
        raise

    except Exception as e:
        logger.error(f"Unexpected error in scrape_url for {url}: {str(e)}")
        return {"url": url, "status": "failed_error", "html": None, "screenshot": None}


async def ascrape_urls(
    urls: List[str],
    num_parallel: int = 10,
    limit: Optional[int] = None,
    show_progress: bool = False,
    timeout: Optional[float] = 10000,
    max_retries: int = 1,
    with_screenshot: bool = True,
    scroll_strategy: ScrollStrategy = "until_stable",
    scroll_max_attempts: int = 15,
    scroll_delay_ms: int = 1400,
    scroll_mode: ScrollMode = "increment",
    headless: bool = True,
    wait_for_js: bool = True,
    use_cache: bool = False,
    context: Optional[BrowserContext] = None,
) -> AsyncIterator[ScrapeResult]:
    """
    Asynchronously scrape a list of URLs with parallel execution.

    Opens one tab per URL (bounded by num_parallel). If `context` is provided,
    it is reused as-is (no browser is launched or closed by this function).
    If `context` is None, a browser/context is launched for this call only
    and torn down before returning.

    Args:
        urls: List of URLs to scrape.
        num_parallel: Number of parallel browser tasks (default: 10).
        limit: Maximum number of successful scrapes to return (None for no limit).
        show_progress: Whether to show a progress bar.
        timeout: Timeout for page navigation in milliseconds.
        max_retries: Number of retries for failed attempts.
        with_screenshot: Whether to capture screenshots.
        scroll_strategy: How to perform scroll ("none", "once", "until_stable").
        scroll_max_attempts: Max scroll attempts if strategy is not "none".
        scroll_delay_ms: Delay between scrolls in ms, if strategy is not "none".
        scroll_mode: Whether to scroll by "jump"ing to bottom or "increment" via viewport height.
        headless: Whether to run browser in headless mode.
        wait_for_js: Whether to wait for JS rendering.
        use_cache: Whether to use Redis caching.
        context: Optional existing BrowserContext to reuse. When provided, no browser
            is launched or closed by this call.

    Yields:
        ScrapeResult: Dictionary containing URL, status, HTML content, and optional screenshot.
    """
    semaphore = asyncio.Semaphore(num_parallel)
    completed_count = 0
    tasks = []

    def yield_cached(url: str, cached_data: dict) -> ScrapeResult:
        screenshot = None
        if with_screenshot and "screenshot" in cached_data:
            try:
                screenshot = base64.b64decode(cached_data["screenshot"])
            except Exception as e:
                logger.error(f"Failed to decode cached screenshot for {url}: {e}")
        logger.debug(f"Cache hit for {url}")
        return {
            "url": url,
            "status": "completed",
            "html": cached_data["content"],
            "screenshot": screenshot,
        }

    async def sem_fetch_and_yield(
        url: str, context: BrowserContext, pbar=None
    ) -> List[ScrapeResult]:
        results = []
        results.append(
            {"url": url, "status": "started", "html": None, "screenshot": None}
        )
        async with semaphore:
            try:
                result = await _scrape_single_url(
                    context,
                    url,
                    timeout=timeout,
                    max_retries=max_retries,
                    with_screenshot=with_screenshot,
                    wait_for_js=wait_for_js,
                    use_cache=False,  # Cache already checked above
                    scroll_strategy=scroll_strategy,
                    scroll_max_attempts=scroll_max_attempts,
                    scroll_delay_ms=scroll_delay_ms,
                    scroll_mode=scroll_mode,
                )
                if pbar:
                    pbar.update(1)
                results.append(result)
            except Exception as e:
                logger.error(f"Exception while scraping {url}: {str(e)}")
                if pbar:
                    pbar.update(1)
                results.append(
                    {
                        "url": url,
                        "status": "failed_error",
                        "html": None,
                        "screenshot": None,
                    }
                )
        return results

    owns_context = context is None
    playwright_instance = None
    browser = None

    try:
        if owns_context:
            playwright_instance = await async_playwright().start()
            ua = UserAgent()
            traces_dir = f"{get_entry_file_dir()}/playwright/traces"
            os.makedirs(traces_dir, exist_ok=True)
            browser = await playwright_instance.chromium.launch(
                headless=headless,
                executable_path=PLAYWRIGHT_CHROMIUM_EXECUTABLE,
                traces_dir=traces_dir,
            )
            context = await browser.new_context(user_agent=ua.random)
            logger.debug("ascrape_urls: launched a fresh browser/context (not reused)")
        else:
            logger.debug("ascrape_urls: reusing externally-provided browser context")

        urls_to_scrape = []
        for url in urls:
            if use_cache:
                cache_key = f"html:{url}"
                cached = cache.get(cache_key)
                if cached:
                    yield yield_cached(url, cached)
                    completed_count += 1
                    if limit and completed_count >= limit:
                        return
                    continue
            urls_to_scrape.append(url)

        if not urls_to_scrape:
            return

        desc = f"Scraping URLs ({num_parallel} max active)"
        if show_progress:
            with tqdm_asyncio(
                total=len(urls_to_scrape),
                desc=desc,
                file=sys.stdout,
                mininterval=0.1,
            ) as pbar:
                tasks = [
                    asyncio.create_task(sem_fetch_and_yield(url, context, pbar))
                    for url in urls_to_scrape
                ]
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
            tasks = [
                asyncio.create_task(sem_fetch_and_yield(url, context, None))
                for url in urls_to_scrape
            ]
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
        if owns_context:
            if context:
                try:
                    await asyncio.wait_for(context.close(), timeout=5.0)
                except Exception as e:
                    logger.debug(f"Error closing self-owned context: {e}")
            if browser:
                try:
                    await asyncio.wait_for(browser.close(), timeout=5.0)
                except Exception as e:
                    logger.debug(f"Error closing self-owned browser: {e}")
            if playwright_instance:
                try:
                    await playwright_instance.stop()
                except Exception as e:
                    logger.debug(f"Error stopping self-owned playwright instance: {e}")
        else:
            logger.debug("ascrape_urls: leaving externally-provided context open")


def scrape_urls(
    urls: List[str],
    num_parallel: int = 10,
    limit: Optional[int] = None,
    show_progress: bool = False,
    timeout: Optional[float] = 10000,
    max_retries: int = 1,
    with_screenshot: bool = True,
    scroll_strategy: ScrollStrategy = "until_stable",
    scroll_max_attempts: int = 15,
    scroll_delay_ms: int = 1400,
    scroll_mode: ScrollMode = "increment",
    headless: bool = True,
    wait_for_js: bool = True,
    use_cache: bool = False,
    context: Optional[BrowserContext] = None,
) -> Iterator[ScrapeResult]:
    """
    Synchronously scrape a list of URLs using Playwright with parallel execution.

    This is a synchronous wrapper around ascrape_urls that manages the event loop.

    Args:
        urls: List of URLs to scrape.
        num_parallel: Number of parallel browser tasks (default: 10).
        limit: Maximum number of successful scrapes to return (None for no limit).
        show_progress: Whether to show a progress bar.
        timeout: Timeout for page navigation in milliseconds.
        max_retries: Number of retries for failed attempts.
        with_screenshot: Whether to capture screenshots.
        scroll_strategy: How to perform scroll ("none", "once", "until_stable").
        scroll_max_attempts: Max scroll attempts if strategy is not "none".
        scroll_delay_ms: Delay between scrolls in ms, if strategy is not "none".
        scroll_mode: Whether to scroll by "jump"ing to bottom or "increment" via viewport height.
        headless: Whether to run browser in headless mode.
        wait_for_js: Whether to wait for JS rendering.
        use_cache: Whether to use Redis caching.
        context: Optional existing BrowserContext to reuse. When provided, no browser
            is launched or closed by this call.

    Yields:
        ScrapeResult: Dictionary containing URL, status, HTML content, and optional screenshot.
    """

    async def run_scrape() -> List[ScrapeResult]:
        results = []
        async for result in ascrape_urls(
            urls=urls,
            num_parallel=num_parallel,
            limit=limit,
            show_progress=show_progress,
            timeout=timeout,
            max_retries=max_retries,
            with_screenshot=with_screenshot,
            scroll_strategy=scroll_strategy,
            scroll_max_attempts=scroll_max_attempts,
            scroll_delay_ms=scroll_delay_ms,
            scroll_mode=scroll_mode,
            headless=headless,
            wait_for_js=wait_for_js,
            use_cache=use_cache,
            context=context,
        ):
            results.append(result)
        return results

    loop = asyncio.get_event_loop()
    if loop.is_running():
        logger.warning(
            "Cannot run synchronous scraping in an already running event loop."
        )
        raise RuntimeError("Synchronous scraping requires a non-running event loop.")

    try:
        results = loop.run_until_complete(run_scrape())
        for result in results:
            yield result
    except Exception as e:
        logger.error(f"Error in synchronous URL scraping: {str(e)}")
        raise


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
