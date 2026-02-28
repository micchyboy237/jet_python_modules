import asyncio
import base64
import os
import sys
from typing import AsyncIterator, Iterator, List, Literal, Optional, TypedDict

from fake_useragent import UserAgent
from jet.cache.redis.types import RedisConfigParams
from jet.cache.redis.utils import RedisCache
from jet.logger import logger
from jet.scrapers.browser.config import PLAYWRIGHT_CHROMIUM_EXECUTABLE
from jet.utils.inspect_utils import get_entry_file_dir
from playwright.async_api import BrowserContext, async_playwright
from tqdm.asyncio import tqdm_asyncio

ScrapeStatus = Literal["started", "completed", "failed_no_html", "failed_error"]
ScrollStrategy = Literal["none", "once", "until_stable"]
ScrollMode = Literal["jump", "increment"]


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
    *,
    scroll_strategy: ScrollStrategy = "none",
    scroll_max_attempts: int = 15,
    scroll_delay_ms: int = 1400,
    scroll_timeout_ms: int = 45000,
    timeout: Optional[float] = 10000,
    max_retries: int = 1,
    with_screenshot: bool = True,
    scroll_mode: ScrollMode = "jump",
    wait_for_js: bool = True,
    use_cache: bool = True,  # Now respected properly
) -> ScrapeResult:
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
                    await page.wait_for_timeout(
                        3500
                    )  # slightly shorter → scrolling takes time

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
                    # Use a cancellation-safe delay
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


async def scroll_to_bottom(
    page,
    *,
    strategy: ScrollStrategy = "until_stable",
    mode: ScrollMode = "jump",
    max_attempts: int = 15,
    delay_ms: int = 1400,
    overall_timeout_ms: int = 45000,
) -> None:
    if strategy == "none":
        return

    logger.debug(f"Starting scroll_to_bottom ({strategy}, mode={mode})")
    start_time = asyncio.get_event_loop().time()

    last_height = await page.evaluate("document.documentElement.scrollHeight")
    STABILIZE_THRESHOLD_PX = 150  # increased slightly for safety

    no_change_count = 0
    MAX_NO_CHANGE = (
        4  # if height doesn't change for this many attempts, consider possibly stuck
    )

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
        else:  # increment
            await page.evaluate(
                "window.scrollBy(0, window.innerHeight * 0.9)"
            )  # 90% to be more conservative

        await page.wait_for_timeout(delay_ms)

        if strategy == "once":
            break

        # Get current metrics
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

        # For increment mode: continue until near bottom, even if height didn't change yet
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
        else:  # jump mode: classic height stability
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


async def scrape_urls(
    urls: List[str],
    num_parallel: int = 10,
    limit: Optional[int] = None,
    show_progress: bool = False,
    timeout: Optional[float] = 10000,
    max_retries: int = 1,
    with_screenshot: bool = True,
    scroll_strategy: ScrollStrategy = "none",
    scroll_max_attempts: int = 15,
    scroll_delay_ms: int = 1400,
    scroll_mode: ScrollMode = "jump",
    headless: bool = True,
    wait_for_js: bool = True,
    use_cache: bool = True,
) -> AsyncIterator[ScrapeResult]:
    semaphore = asyncio.Semaphore(num_parallel)
    completed_count = 0
    tasks = []

    # Helper to yield cached result immediately
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
                result = await scrape_url(
                    context,
                    url,
                    timeout=timeout,
                    max_retries=max_retries,
                    with_screenshot=with_screenshot,
                    wait_for_js=wait_for_js,
                    use_cache=False,  # Important: disable inner cache check
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
    scroll_strategy: ScrollStrategy = "none",
    scroll_max_attempts: int = 15,
    scroll_delay_ms: int = 1400,
    scroll_mode: ScrollMode = "jump",
    headless: bool = True,
    wait_for_js: bool = True,
    use_cache: bool = True,
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
        scroll_strategy: How to perform scroll ("none", "once", "until_stable").
        scroll_max_attempts: Max scroll attempts if strategy is not "none".
        scroll_delay_ms: Delay between scrolls in ms, if strategy is not "none".
        scroll_mode: Whether to scroll by "jump"ing to bottom or "increment" via viewport height.
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
            scroll_strategy=scroll_strategy,
            scroll_max_attempts=scroll_max_attempts,
            scroll_delay_ms=scroll_delay_ms,
            scroll_mode=scroll_mode,
            headless=headless,
            wait_for_js=wait_for_js,
            use_cache=use_cache,
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
