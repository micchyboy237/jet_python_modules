import asyncio
import os
from typing import List, Optional, TypedDict

from jet.cache.redis.types import RedisConfigParams
from jet.cache.redis.utils import RedisCache
from jet.logger import logger
from jet.scrapers.browser.config import get_browser_config
from jet.utils.inspect_utils import get_entry_file_dir
from playwright.async_api import Browser as AsyncBrowser
from playwright.async_api import Page as AsyncPage
from playwright.async_api import async_playwright
from playwright.sync_api import Browser as SyncBrowser
from playwright.sync_api import Page as SyncPage
from playwright.sync_api import sync_playwright
from playwright_stealth import Stealth  # ✅ v2.x API — no stealth_sync/stealth_async
from tqdm.asyncio import tqdm

GENERATED_DIR = (
    "/Users/jethroestrada/Desktop/External_Projects/Jet_Apps/my-jobs/generated"
)
os.makedirs(GENERATED_DIR, exist_ok=True)

REDIS_CONFIG = RedisConfigParams(port=6379)

browser_page = None

# Custom init script for patches NOT covered by playwright-stealth v2.x
CUSTOM_STEALTH_INIT_SCRIPT = """
    // playwright-stealth v2.x disables chrome.runtime by default;
    // these cover remaining gaps specific to Playwright automation
    delete window.__playwright;
    delete window.__pw_manual;
    Object.defineProperty(navigator, 'plugins', {
        get: () => [1, 2, 3, 4, 5]
    });
"""


class PageDimensions(TypedDict):
    width: int
    height: int
    deviceScaleFactor: float


class PageContent(TypedDict):
    url: str
    dimensions: PageDimensions
    screenshot: str
    html: str


# ---------------------------------------------------------------------------
# Stealth instance (reused across sessions, configured once)
# ---------------------------------------------------------------------------

_stealth = Stealth(
    navigator_languages_override=("en-PH", "en-US", "en"),
    chrome_runtime=False,  # Disabled by default in v2.x; can cause site breakage
)


# ---------------------------------------------------------------------------
# Session Setup
# ---------------------------------------------------------------------------


def setup_sync_browser_session(*, headless: bool = False) -> SyncBrowser:
    """Create a synchronous Playwright session with dynamic config + stealth."""
    config = get_browser_config()

    effective_headless = headless
    if config.source == "system_chrome" and headless:
        logger.warning(
            "System Chrome does not support legacy headless. Forcing headed mode."
        )
        effective_headless = False

    pw = sync_playwright().start()
    traces_dir = f"{get_entry_file_dir()}/playwright/traces"
    os.makedirs(traces_dir, exist_ok=True)

    launch_kwargs = dict(
        headless=effective_headless,
        traces_dir=traces_dir,
        user_agent=config.user_agent,
        locale=config.locale,
        timezone_id=config.timezone_id,
        java_script_enabled=True,
        bypass_csp=True,
        viewport={
            "width": config.viewport_width,
            "height": config.viewport_height,
        },
        extra_http_headers=config.extra_http_headers,
    )

    if config.channel:
        launch_kwargs["channel"] = config.channel
    if config.executable_path:
        launch_kwargs["executable_path"] = config.executable_path

    context = pw.chromium.launch_persistent_context(
        user_data_dir=os.path.join(GENERATED_DIR, "browser_context"),
        **launch_kwargs,
    )

    # ✅ v2.x: apply stealth to the entire context (all future pages inherit it)
    _stealth.apply_stealth_sync(context)

    # Wrap new_page to inject custom patches that playwright-stealth doesn't cover
    original_new_page = context.new_page

    def patched_new_page(*args, **kwargs):
        page = original_new_page(*args, **kwargs)
        page.add_init_script(CUSTOM_STEALTH_INIT_SCRIPT)
        return page

    context.new_page = patched_new_page

    logger.debug(
        f"Sync browser session created: source={config.source}, "
        f"headless={effective_headless}"
    )
    return context


async def setup_async_browser_session(*, headless: bool = False) -> AsyncBrowser:
    """Create an asynchronous Playwright session with dynamic config + stealth."""
    config = get_browser_config()

    effective_headless = headless
    if config.source == "system_chrome" and headless:
        logger.warning(
            "System Chrome does not support legacy headless. Forcing headed mode."
        )
        effective_headless = False

    pw = await async_playwright().start()

    launch_kwargs = dict(
        headless=effective_headless,
        user_agent=config.user_agent,
        locale=config.locale,
        timezone_id=config.timezone_id,
        java_script_enabled=True,
        bypass_csp=True,
        viewport={
            "width": config.viewport_width,
            "height": config.viewport_height,
        },
        extra_http_headers=config.extra_http_headers,
    )

    if config.channel:
        launch_kwargs["channel"] = config.channel
    if config.executable_path:
        launch_kwargs["executable_path"] = config.executable_path

    context = await pw.chromium.launch_persistent_context(
        user_data_dir=os.path.join(GENERATED_DIR, "browser_context"),
        **launch_kwargs,
    )

    # ✅ v2.x: apply stealth to the entire context
    await _stealth.apply_stealth_async(context)

    original_new_page = context.new_page

    async def patched_new_page(*args, **kwargs):
        page = await original_new_page(*args, **kwargs)
        await page.add_init_script(CUSTOM_STEALTH_INIT_SCRIPT)
        return page

    context.new_page = patched_new_page

    logger.debug(
        f"Async browser session created: source={config.source}, "
        f"headless={effective_headless}"
    )
    return context


# ---------------------------------------------------------------------------
# Page Helpers
# ---------------------------------------------------------------------------


def setup_browser_page(page: Optional[SyncPage] = None, headless: bool = True):
    global browser_page
    if not browser_page:
        browser_page = page or setup_sync_browser_page(headless=headless)
    return browser_page


async def asetup_browser_page(*, headless: bool = True) -> AsyncPage:
    browser = await setup_async_browser_session(headless=headless)
    return await browser.new_page()


def setup_sync_browser_page(*, headless: bool = False) -> SyncPage:
    browser = setup_sync_browser_session(headless=headless)
    return browser.new_page()


async def setup_async_browser_page(*, headless: bool = False) -> AsyncPage:
    browser = await setup_async_browser_session(headless=headless)
    return await browser.new_page()


# ---------------------------------------------------------------------------
# Content Fetching
# ---------------------------------------------------------------------------


def fetch_page_content_sync(
    url: str,
    wait_for_css: Optional[List[str]],
    max_wait_timeout: int = 10000,
    headless: bool = True,
    use_cache: bool = False,
) -> PageContent:
    cache = RedisCache(config=REDIS_CONFIG)
    cache_key = url
    cached_result = cache.get(cache_key) if use_cache else None
    bp = setup_browser_page(headless=headless)

    if cached_result:
        logger.log(
            "scrape_url: Cache hit for", cache_key, colors=["LOG", "BRIGHT_SUCCESS"]
        )
        return cached_result

    if wait_for_css:
        logger.log("Waiting for elements css:", wait_for_css, colors=["GRAY", "DEBUG"])
        for css_selector in wait_for_css:
            bp.wait_for_selector(css_selector, timeout=max_wait_timeout)

    screenshot_path = f"{GENERATED_DIR}/example.png"
    bp.screenshot(path=screenshot_path)

    dimensions: PageDimensions = bp.evaluate(
        """() => ({
        width: document.documentElement.clientWidth,
        height: document.documentElement.clientHeight,
        deviceScaleFactor: window.devicePixelRatio
    })"""
    )

    result: PageContent = {
        "url": url,
        "dimensions": dimensions,
        "screenshot": os.path.realpath(screenshot_path),
        "html": bp.content(),
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
    cache = RedisCache(config=REDIS_CONFIG)
    cache_key = url
    cached_result = cache.get(cache_key) if use_cache else None
    bp = page or await asetup_browser_page(headless=headless)

    try:
        if cached_result:
            logger.log(
                "scrape_url: Cache hit for",
                cache_key,
                colors=["LOG", "BRIGHT_SUCCESS"],
            )
            return cached_result

        if wait_for_css:
            logger.log(
                "Waiting for elements css:", wait_for_css, colors=["GRAY", "DEBUG"]
            )
            for css_selector in wait_for_css:
                await bp.wait_for_selector(css_selector, timeout=max_wait_timeout)

        screenshot_path = f"{GENERATED_DIR}/example.png"
        await bp.screenshot(path=screenshot_path)

        dimensions: PageDimensions = await bp.evaluate(
            """() => ({
            width: document.documentElement.clientWidth,
            height: document.documentElement.clientHeight,
            deviceScaleFactor: window.devicePixelRatio
        })"""
        )

        result: PageContent = {
            "url": url,
            "dimensions": dimensions,
            "screenshot": os.path.realpath(screenshot_path),
            "html": await bp.content(),
        }

        if use_cache:
            cache.set(cache_key, result)
        return result
    finally:
        if not page:
            await bp.close()


# ---------------------------------------------------------------------------
# Scraping
# ---------------------------------------------------------------------------


def scrape_sync(
    url: str,
    wait_for_css: Optional[List[str]] = None,
    headless: bool = True,
    use_cache: bool = False,
) -> PageContent:
    bp = setup_browser_page(headless=headless)
    bp.goto(url, wait_until="domcontentloaded")
    return fetch_page_content_sync(url, wait_for_css, use_cache=use_cache)


async def scrape_async(
    url: str,
    wait_for_css: Optional[List[str]] = None,
    headless: bool = True,
    use_cache: bool = False,
) -> PageContent:
    bp = await asetup_browser_page(headless=headless)
    try:
        await bp.goto(url)
        return await fetch_page_content_async(
            url, wait_for_css, page=bp, use_cache=use_cache
        )
    finally:
        await bp.close()


async def setup_browser_pool(
    max_pages: int = 2, headless: bool = False
) -> List[AsyncPage]:
    browser = await setup_async_browser_session(headless=headless)
    return [await browser.new_page() for _ in range(max_pages)]


async def scrape_async_limited(
    urls: List[str], max_concurrent_tasks: int = 2, headless: bool = False
) -> List[PageContent]:
    pages = await setup_browser_pool(max_concurrent_tasks, headless)
    page_queue = asyncio.Queue()
    for page in pages:
        await page_queue.put(page)

    results = []
    progress_bar = tqdm(total=len(urls), desc="Scraping Progress", unit="url")

    async def bound_scrape(url) -> PageContent:
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
