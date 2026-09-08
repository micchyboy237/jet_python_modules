"""
Unified Playwright service: browser setup, stealth, scraping, scrolling,
caching, and batch processing.

All paths, cache settings, and tuning parameters are configurable via
ScraperSettings. No hardcoded user-specific paths remain.
"""

import asyncio
import base64
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Iterator, List, Literal, Optional, TypedDict, Union

from jet.cache.redis.types import RedisConfigParams
from jet.cache.redis.utils import RedisCache
from jet.logger import logger
from jet.scrapers.browser.config import get_browser_config
from playwright.async_api import BrowserContext as AsyncBrowserContext
from playwright.async_api import Page as AsyncPage
from playwright.async_api import async_playwright
from playwright.sync_api import BrowserContext as SyncBrowserContext
from playwright.sync_api import sync_playwright
from playwright_stealth import Stealth
from tqdm.asyncio import tqdm_asyncio

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
ScrapeStatus = Literal["started", "completed", "failed_no_html", "failed_error"]
ScrollStrategy = Literal["none", "once", "until_stable"]
ScrollMode = Literal["jump", "increment"]


class PageDimensions(TypedDict):
    width: int
    height: int
    deviceScaleFactor: float


class ScrapeResult(TypedDict):
    url: str
    status: ScrapeStatus
    html: Optional[str]
    screenshot: Optional[bytes]
    dimensions: Optional[PageDimensions]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def _default_output_dir() -> Path:
    """Derive output directory from env var or platform-appropriate default."""
    env = os.environ.get("JET_SCRAPER_OUTPUT_DIR")
    if env:
        return Path(env)
    xdg_data = os.environ.get("XDG_DATA_HOME")
    if xdg_data:
        return Path(xdg_data) / "jet_scraper"
    if os.name == "nt":
        return Path(os.environ.get("LOCALAPPDATA", "~")) / "JetScraper"
    return Path.home() / ".local" / "share" / "jet_scraper"


@dataclass(frozen=True)
class ScraperSettings:
    """Immutable configuration for all scraper operations.

    Attributes:
        output_dir:           Root directory for generated files, contexts, traces.
        context_subdir:       Subdirectory name under output_dir for persistent browser data.
        traces_subdir:        Subdirectory name under output_dir for Playwright traces.
        redis_config:         Redis connection parameters for caching.
        cache_ttl:            Time-to-live in seconds for cached scrape results.
        cache_key_prefix:     Prefix for Redis cache keys (avoids collisions across projects).
        scroll_threshold_px:  Pixel tolerance for considering scroll "at bottom".
        scroll_max_no_change: Consecutive no-change scroll attempts before stopping.
        stealth_script:       Custom JS injected into every new page.
    """

    output_dir: Path = field(default_factory=_default_output_dir)
    context_subdir: str = "browser_context"
    traces_subdir: str = "traces"
    redis_config: RedisConfigParams = field(
        default_factory=lambda: RedisConfigParams(port=6379)
    )
    cache_ttl: int = 3600
    cache_key_prefix: str = "scrape"
    scroll_threshold_px: int = 150
    scroll_max_no_change: int = 4
    stealth_script: str = (
        "delete window.__playwright;"
        "delete window.__pw_manual;"
        "Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });"
    )

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / self.context_subdir).mkdir(parents=True, exist_ok=True)
        (self.output_dir / self.traces_subdir).mkdir(parents=True, exist_ok=True)

    @property
    def context_dir(self) -> Path:
        return self.output_dir / self.context_subdir

    @property
    def traces_dir(self) -> Path:
        return self.output_dir / self.traces_subdir

    def cache_key(self, url: str) -> str:
        return f"{self.cache_key_prefix}:{url}"


DEFAULT_SETTINGS = ScraperSettings()


# ---------------------------------------------------------------------------
# Cache Helpers
# ---------------------------------------------------------------------------
def _decode_cached_screenshot(data: dict, with_screenshot: bool) -> Optional[bytes]:
    if not with_screenshot or not data.get("screenshot"):
        return None
    try:
        return base64.b64decode(data["screenshot"])
    except Exception as e:
        logger.error(f"Failed to decode cached screenshot: {e}")
        return None


def _cache_read(
    settings: ScraperSettings, url: str, with_screenshot: bool
) -> Optional[ScrapeResult]:
    cache = RedisCache(config=settings.redis_config)
    cached = cache.get(settings.cache_key(url))
    if not cached:
        return None
    logger.debug(f"Cache hit for {url}")
    return {
        "url": url,
        "status": "completed",
        "html": cached["html"],
        "screenshot": _decode_cached_screenshot(cached, with_screenshot),
        "dimensions": cached.get("dimensions"),
    }


def _cache_write(
    settings: ScraperSettings,
    url: str,
    html: str,
    dimensions: PageDimensions,
    screenshot: Optional[bytes],
) -> None:
    cache = RedisCache(config=settings.redis_config)
    data: dict = {"html": html, "dimensions": dimensions}
    if screenshot:
        data["screenshot"] = base64.b64encode(screenshot).decode()
    cache.set(settings.cache_key(url), data, ttl=settings.cache_ttl)


# ---------------------------------------------------------------------------
# Result Builder & JS Snippets
# ---------------------------------------------------------------------------
def _build_result(
    url: str,
    status: ScrapeStatus,
    html: Optional[str] = None,
    screenshot: Optional[bytes] = None,
    dimensions: Optional[PageDimensions] = None,
) -> ScrapeResult:
    return {
        "url": url,
        "status": status,
        "html": html,
        "screenshot": screenshot,
        "dimensions": dimensions,
    }


_DIMENSIONS_JS = """() => ({
    width: document.documentElement.clientWidth,
    height: document.documentElement.clientHeight,
    deviceScaleFactor: window.devicePixelRatio
})"""

_SCROLL_METRICS_JS = """() => ({
    scrollHeight: document.documentElement.scrollHeight,
    scrollY: window.scrollY,
    innerHeight: window.innerHeight
})"""


# ---------------------------------------------------------------------------
# Browser Session Factories
# ---------------------------------------------------------------------------
def _build_launch_kwargs(settings: ScraperSettings, headless: bool) -> dict:
    config = get_browser_config()
    effective_headless = headless
    if config.source == "system_chrome" and headless:
        logger.warning(
            "System Chrome does not support legacy headless. Forcing headed."
        )
        effective_headless = False

    kwargs = dict(
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
        kwargs["channel"] = config.channel
    if config.executable_path:
        kwargs["executable_path"] = config.executable_path
    return kwargs


def _patch_new_page(
    context: Union[SyncBrowserContext, AsyncBrowserContext],
    stealth: Stealth,
    script: str,
) -> None:
    """Apply stealth and inject init script for both sync and async contexts.

    Uses isinstance against Playwright's AsyncBrowserContext to reliably
    distinguish sync from async. Previous hasattr('_async') heuristic was
    incorrect and caused coroutines to be used without await.
    """
    original_new_page = context.new_page

    if isinstance(context, AsyncBrowserContext):

        async def patched_async(*args, **kwargs):
            page = await original_new_page(*args, **kwargs)
            await page.add_init_script(script)
            return page

        context.new_page = patched_async  # type: ignore[assignment]
    else:

        def patched_sync(*args, **kwargs):
            page = original_new_page(*args, **kwargs)
            page.add_init_script(script)
            return page

        context.new_page = patched_sync  # type: ignore[assignment]


def create_sync_context(
    *,
    headless: bool = False,
    settings: Optional[ScraperSettings] = None,
) -> SyncBrowserContext:
    """Create a synchronous persistent browser context with stealth."""
    s = settings or DEFAULT_SETTINGS
    pw = sync_playwright().start()
    kwargs = _build_launch_kwargs(s, headless)
    kwargs["traces_dir"] = str(s.traces_dir)
    context = pw.chromium.launch_persistent_context(
        user_data_dir=str(s.context_dir), **kwargs
    )
    stealth = Stealth(
        navigator_languages_override=("en-PH", "en-US", "en"), chrome_runtime=False
    )
    stealth.apply_stealth_sync(context)
    _patch_new_page(context, stealth, s.stealth_script)
    logger.debug(
        f"Sync context created: output={s.output_dir}, headless={kwargs['headless']}"
    )
    return context


async def create_async_context(
    *,
    headless: bool = False,
    settings: Optional[ScraperSettings] = None,
) -> AsyncBrowserContext:
    """Create an asynchronous persistent browser context with stealth."""
    s = settings or DEFAULT_SETTINGS
    pw = await async_playwright().start()
    kwargs = _build_launch_kwargs(s, headless)
    context = await pw.chromium.launch_persistent_context(
        user_data_dir=str(s.context_dir), **kwargs
    )
    stealth = Stealth(
        navigator_languages_override=("en-PH", "en-US", "en"), chrome_runtime=False
    )
    await stealth.apply_stealth_async(context)
    _patch_new_page(context, stealth, s.stealth_script)
    logger.debug(
        f"Async context created: output={s.output_dir}, headless={kwargs['headless']}"
    )
    return context


# ---------------------------------------------------------------------------
# Scroll Logic
# ---------------------------------------------------------------------------
async def scroll_to_bottom(
    page: AsyncPage,
    *,
    strategy: ScrollStrategy = "until_stable",
    mode: ScrollMode = "increment",
    max_attempts: int = 15,
    delay_ms: int = 1400,
    overall_timeout_ms: int = 45000,
    settings: Optional[ScraperSettings] = None,
) -> None:
    """Scroll page to bottom using configurable strategy."""
    if strategy == "none":
        return

    s = settings or DEFAULT_SETTINGS
    logger.debug(f"scroll_to_bottom: strategy={strategy}, mode={mode}")

    # Wait for initial dynamic content to render before measuring scroll height.
    # Without this, SPAs that load content via XHR after domcontentloaded will
    # report scrollHeight == innerHeight on the first check, causing a false
    # "bottom reached" before any scrolling occurs.
    await page.wait_for_timeout(delay_ms)

    start_time = asyncio.get_event_loop().time()
    last_height = await page.evaluate("document.documentElement.scrollHeight")
    no_change_count = 0

    for attempt in range(1, max_attempts + 1):
        elapsed_ms = (asyncio.get_event_loop().time() - start_time) * 1000
        if overall_timeout_ms and elapsed_ms > overall_timeout_ms:
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

        metrics = await page.evaluate(_SCROLL_METRICS_JS)
        new_height = metrics["scrollHeight"]
        view_bottom = metrics["scrollY"] + metrics["innerHeight"]
        bottom_reached = view_bottom >= (new_height - s.scroll_threshold_px)
        height_changed = new_height != last_height
        no_change_count = 0 if height_changed else no_change_count + 1

        if mode == "increment":
            if bottom_reached and not height_changed:
                # Require minimum scroll activity before accepting stability.
                # Prevents false positives on pages where initial scrollHeight
                # equals viewport height before dynamic content loads.
                if attempt >= 2 or height_changed:
                    logger.debug(f"Bottom reached and stable after {attempt} attempts")
                    break
                logger.debug("Apparent bottom on first attempt, continuing to verify")
            if no_change_count >= s.scroll_max_no_change:
                logger.warning(
                    f"No height change for {s.scroll_max_no_change} attempts, stopping"
                )
                break
        elif not height_changed:
            if attempt >= 2:
                logger.debug(f"Scroll stabilized after {attempt} attempts")
                break

        last_height = new_height


# ---------------------------------------------------------------------------
# Core Scrape Logic
# ---------------------------------------------------------------------------
async def _scrape_core_async(
    context: AsyncBrowserContext,
    url: str,
    settings: ScraperSettings,
    *,
    scroll_strategy: ScrollStrategy,
    scroll_max_attempts: int,
    scroll_delay_ms: int,
    scroll_timeout_ms: int,
    scroll_mode: ScrollMode,
    timeout: float,
    max_retries: int,
    with_screenshot: bool,
    wait_for_js: bool,
    use_cache: bool,
) -> ScrapeResult:
    """Internal async scrape implementation. Single source of truth."""
    if use_cache:
        cached = _cache_read(settings, url, with_screenshot)
        if cached:
            return cached

    for attempt in range(max_retries + 1):
        page = None
        try:
            page = await context.new_page()
            logger.debug(f"Navigating to {url} (attempt {attempt + 1})")
            await page.goto(url, timeout=timeout, wait_until="domcontentloaded")

            if wait_for_js:
                await page.wait_for_timeout(3500)

            await scroll_to_bottom(
                page,
                strategy=scroll_strategy,
                max_attempts=scroll_max_attempts,
                delay_ms=scroll_delay_ms,
                overall_timeout_ms=scroll_timeout_ms,
                mode=scroll_mode,
                settings=settings,
            )

            html = await page.content()
            screenshot = (
                await page.screenshot(full_page=True) if with_screenshot else None
            )
            dimensions = await page.evaluate(_DIMENSIONS_JS)

            if use_cache:
                _cache_write(settings, url, html, dimensions, screenshot)

            return _build_result(url, "completed", html, screenshot, dimensions)

        except Exception as e:
            logger.error(
                f"Error scraping {url}: {e} (attempt {attempt + 1}/{max_retries + 1})"
            )
            if attempt < max_retries:
                try:
                    await asyncio.wait_for(
                        asyncio.sleep(2 ** (attempt + 1)), timeout=10.0
                    )
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    break
        finally:
            if page:
                try:
                    await asyncio.wait_for(page.close(), timeout=5.0)
                except Exception:
                    pass

    return _build_result(url, "failed_no_html")


def _scrape_core_sync(
    context: SyncBrowserContext,
    url: str,
    settings: ScraperSettings,
    *,
    timeout: float,
    max_retries: int,
    with_screenshot: bool,
    wait_for_js: bool,
    use_cache: bool,
) -> ScrapeResult:
    """Internal sync scrape. Mirrors async core but uses native sync Playwright API."""
    if use_cache:
        cached = _cache_read(settings, url, with_screenshot)
        if cached:
            return cached

    for attempt in range(max_retries + 1):
        page = None
        try:
            page = context.new_page()
            logger.debug(f"[sync] Navigating to {url} (attempt {attempt + 1})")
            page.goto(url, timeout=timeout, wait_until="domcontentloaded")

            if wait_for_js:
                page.wait_for_timeout(3500)

            html = page.content()
            screenshot = page.screenshot(full_page=True) if with_screenshot else None
            dimensions = page.evaluate(_DIMENSIONS_JS)

            if use_cache:
                _cache_write(settings, url, html, dimensions, screenshot)

            return _build_result(url, "completed", html, screenshot, dimensions)

        except Exception as e:
            logger.error(
                f"[sync] Error scraping {url}: {e} (attempt {attempt + 1}/{max_retries + 1})"
            )
        finally:
            if page:
                try:
                    page.close()
                except Exception:
                    pass

    return _build_result(url, "failed_no_html")


# ---------------------------------------------------------------------------
# Public Scrape API
# ---------------------------------------------------------------------------
async def scrape_url(
    context: AsyncBrowserContext,
    url: str,
    *,
    scroll_strategy: ScrollStrategy = "until_stable",
    scroll_max_attempts: int = 15,
    scroll_delay_ms: int = 1400,
    scroll_timeout_ms: int = 45000,
    scroll_mode: ScrollMode = "increment",
    timeout: float = 10000,
    max_retries: int = 1,
    with_screenshot: bool = True,
    wait_for_js: bool = False,
    use_cache: bool = False,
    settings: Optional[ScraperSettings] = None,
) -> ScrapeResult:
    """Scrape a single URL asynchronously with retry, scroll, cache, and screenshot."""
    s = settings or DEFAULT_SETTINGS
    return await _scrape_core_async(
        context,
        url,
        s,
        scroll_strategy=scroll_strategy,
        scroll_max_attempts=scroll_max_attempts,
        scroll_delay_ms=scroll_delay_ms,
        scroll_timeout_ms=scroll_timeout_ms,
        scroll_mode=scroll_mode,
        timeout=timeout,
        max_retries=max_retries,
        with_screenshot=with_screenshot,
        wait_for_js=wait_for_js,
        use_cache=use_cache,
    )


def scrape_url_sync(
    context: SyncBrowserContext,
    url: str,
    *,
    timeout: float = 10000,
    max_retries: int = 1,
    with_screenshot: bool = True,
    wait_for_js: bool = False,
    use_cache: bool = False,
    settings: Optional[ScraperSettings] = None,
) -> ScrapeResult:
    """Scrape a single URL synchronously. Scroll strategies not supported in sync mode."""
    s = settings or DEFAULT_SETTINGS
    return _scrape_core_sync(
        context,
        url,
        s,
        timeout=timeout,
        max_retries=max_retries,
        with_screenshot=with_screenshot,
        wait_for_js=wait_for_js,
        use_cache=use_cache,
    )


# ---------------------------------------------------------------------------
# Batch Scraping
# ---------------------------------------------------------------------------
async def scrape_batch(
    urls: List[str],
    *,
    num_parallel: int = 5,
    limit: Optional[int] = None,
    show_progress: bool = True,
    headless: bool = True,
    use_cache: bool = False,
    settings: Optional[ScraperSettings] = None,
    **scrape_kwargs,
) -> AsyncIterator[ScrapeResult]:
    """Async generator yielding ScrapeResults concurrently from a shared context."""
    s = settings or DEFAULT_SETTINGS
    semaphore = asyncio.Semaphore(num_parallel)
    completed = 0
    with_screenshot = scrape_kwargs.get("with_screenshot", True)
    context = await create_async_context(headless=headless, settings=s)

    try:
        pending_urls = []
        for url in urls:
            if use_cache:
                cached = _cache_read(s, url, with_screenshot)
                if cached:
                    yield cached
                    completed += 1
                    if limit and completed >= limit:
                        return
                    continue
            pending_urls.append(url)

        if not pending_urls:
            return

        pbar = tqdm_asyncio(
            total=len(pending_urls), desc="Scraping", disable=not show_progress
        )

        async def _task(u: str) -> ScrapeResult:
            async with semaphore:
                result = await scrape_url(
                    context, u, use_cache=use_cache, settings=s, **scrape_kwargs
                )
                pbar.update(1)
                return result

        tasks = [asyncio.create_task(_task(u)) for u in pending_urls]
        for coro in asyncio.as_completed(tasks):
            result = await coro
            yield result
            completed += 1
            if limit and completed >= limit:
                for t in tasks:
                    if not t.done():
                        t.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                break

        pbar.close()
    finally:
        await context.close()


def scrape_batch_sync(
    urls: List[str],
    *,
    headless: bool = True,
    use_cache: bool = False,
    show_progress: bool = True,
    settings: Optional[ScraperSettings] = None,
    **scrape_kwargs,
) -> Iterator[ScrapeResult]:
    """Synchronous batch scraper using native sync Playwright API."""
    s = settings or DEFAULT_SETTINGS
    with_screenshot = scrape_kwargs.get("with_screenshot", True)
    context = create_sync_context(headless=headless, settings=s)
    try:
        from tqdm import tqdm as sync_tqdm

        url_iter = sync_tqdm(urls, desc="Scraping (sync)", disable=not show_progress)
        for url in url_iter:
            if use_cache:
                cached = _cache_read(s, url, with_screenshot)
                if cached:
                    yield cached
                    continue
            yield scrape_url_sync(
                context, url, use_cache=use_cache, settings=s, **scrape_kwargs
            )
    finally:
        context.close()
