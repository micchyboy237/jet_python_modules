import os
from typing import TypedDict, List, Optional, Union
from jet.logger import logger
from playwright.sync_api import sync_playwright, Browser as SyncBrowser, Page as SyncPage
from playwright.async_api import async_playwright, Browser as AsyncBrowser, Page as AsyncPage

GENERATED_DIR = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/my-jobs/generated"
os.makedirs(GENERATED_DIR, exist_ok=True)


class PageDimensions(TypedDict):
    width: int
    height: int
    deviceScaleFactor: float


class PageContent(TypedDict):
    dimensions: PageDimensions
    screenshot: str
    html: str


class ScrapeHTMLResult(PageContent):
    browser_page: Union[SyncPage, AsyncPage]


def setup_sync_browser_session(*, headless: bool = False) -> SyncBrowser:
    """Sets up a synchronous Playwright browser session and returns the browser instance."""
    playwright = sync_playwright().start()
    browser = playwright.chromium.launch(headless=headless)
    return browser


async def setup_async_browser_session(*, headless: bool = False) -> AsyncBrowser:
    """Sets up an asynchronous Playwright browser session and returns the browser instance."""
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(headless=headless)
    return browser


def setup_sync_browser_page(*, headless: bool = False) -> SyncPage:
    """Sets up a synchronous Playwright browser session and returns the browser instance."""
    browser = setup_sync_browser_session(headless=headless)
    return browser.new_page()


async def setup_async_browser_page(*, headless: bool = False) -> AsyncPage:
    """Sets up an asynchronous Playwright browser page and returns the browser instance."""
    browser = await setup_async_browser_session(headless=headless)
    return await browser.new_page()


def fetch_page_content_sync(browser_page, wait_for_css: Optional[List[str]], max_wait_timeout: int = 10000) -> PageContent:
    """Fetches page content synchronously, including screenshot and HTML."""
    if wait_for_css:
        logger.log("Waiting for elements css:",
                   wait_for_css, colors=["GRAY", "DEBUG"])
        for css_selector in wait_for_css:
            browser_page.wait_for_selector(
                css_selector, timeout=max_wait_timeout)

    screenshot_path = f'{GENERATED_DIR}/example.png'
    browser_page.screenshot(path=screenshot_path)

    dimensions: PageDimensions = browser_page.evaluate('''() => ({
        width: document.documentElement.clientWidth,
        height: document.documentElement.clientHeight,
        deviceScaleFactor: window.devicePixelRatio
    })''')

    return {
        "dimensions": dimensions,
        "screenshot": os.path.realpath(screenshot_path),
        "html": browser_page.content()
    }


async def fetch_page_content_async(browser_page, wait_for_css: Optional[List[str]], max_wait_timeout: int = 10000) -> PageContent:
    """Fetches page content asynchronously, including screenshot and HTML."""
    if wait_for_css:
        logger.log("Waiting for elements css:",
                   wait_for_css, colors=["GRAY", "DEBUG"])
        for css_selector in wait_for_css:
            await browser_page.wait_for_selector(css_selector, timeout=max_wait_timeout)

    screenshot_path = f'{GENERATED_DIR}/example.png'
    await browser_page.screenshot(path=screenshot_path)

    dimensions: PageDimensions = await browser_page.evaluate('''() => ({
        width: document.documentElement.clientWidth,
        height: document.documentElement.clientHeight,
        deviceScaleFactor: window.devicePixelRatio
    })''')

    return {
        "dimensions": dimensions,
        "screenshot": os.path.realpath(screenshot_path),
        "html": await browser_page.content()
    }


def scrape_sync(url: str, wait_for_css: Optional[List[str]] = None, browser_page: Optional[SyncPage] = None) -> ScrapeHTMLResult:
    """Scrapes a webpage synchronously."""
    browser_page = browser_page or setup_sync_browser_page()
    browser_page.goto(url)
    result = fetch_page_content_sync(browser_page, wait_for_css)

    return {
        "browser_page": browser_page,
        **result
    }


async def scrape_async(url: str, wait_for_css: Optional[List[str]] = None, browser_page: Optional[AsyncPage] = None) -> ScrapeHTMLResult:
    """Scrapes a webpage asynchronously."""
    browser_page = browser_page or await setup_async_browser_page()
    await browser_page.goto(url)
    result = await fetch_page_content_async(browser_page, wait_for_css)

    return {
        "browser_page": browser_page,
        **result
    }
