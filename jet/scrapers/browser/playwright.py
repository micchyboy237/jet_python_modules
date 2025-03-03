from tqdm.asyncio import tqdm
import asyncio
import os

from typing import TypedDict, List, Optional, Union
from playwright.sync_api import sync_playwright, Browser as SyncBrowser, Page as SyncPage
from playwright.async_api import async_playwright, Browser as AsyncBrowser, Page as AsyncPage
from jet.logger import logger

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


def scrape_sync(url: str, wait_for_css: Optional[List[str]] = None, browser_page: Optional[SyncPage] = None) -> PageContent:
    """Scrapes a webpage synchronously."""
    browser_page = browser_page or setup_sync_browser_page()
    browser_page.goto(url)
    result = fetch_page_content_sync(browser_page, wait_for_css)

    return result


async def scrape_async(url: str, wait_for_css: Optional[List[str]] = None, browser_page: Optional[AsyncPage] = None) -> PageContent:
    """Scrapes a webpage asynchronously."""
    browser_page = browser_page or await setup_async_browser_page()
    await browser_page.goto(url)
    result = await fetch_page_content_async(browser_page, wait_for_css)

    return result


async def setup_browser_pool(max_pages: int = 2, headless: bool = False) -> List[AsyncPage]:
    """Creates a pool of browser pages to be shared among tasks."""
    browser = await setup_async_browser_session(headless=headless)
    return [await browser.new_page() for _ in range(max_pages)]


async def scrape_async_limited(urls: List[str], max_concurrent_tasks: int = 2, headless: bool = False) -> List[PageContent]:
    """Scrapes multiple URLs asynchronously, limiting concurrent tasks and tracking progress."""

    pages = await setup_browser_pool(max_concurrent_tasks, headless)
    page_queue = asyncio.Queue()

    for page in pages:
        await page_queue.put(page)  # Add pages to queue

    results = []
    progress_bar = tqdm(total=len(urls), desc="Scraping Progress", unit="url")

    async def bound_scrape(url) -> PageContent:
        page = await page_queue.get()  # Get a page from the queue
        try:
            result = await scrape_async(url, browser_page=page)
            results.append(result)
            progress_bar.update(1)  # Update progress
        finally:
            await page_queue.put(page)  # Return the page to the queue
        return result

    await asyncio.gather(*(bound_scrape(url) for url in urls))

    # Close browser pages after scraping
    for page in pages:
        await page.close()

    progress_bar.close()  # Close tqdm progress bar
    return results

# Example Usage
if __name__ == "__main__":
    urls_to_scrape = [
        "https://example.com",
        "https://example.org",
        "https://example.net"
    ]

    asyncio.run(scrape_async_limited(urls_to_scrape))
