import os
from typing import TypedDict, List, Optional
from playwright.sync_api import sync_playwright
from playwright.async_api import async_playwright
from jet.logger.timer import sleep_countdown

GENERATED_DIR = "generated"
os.makedirs(GENERATED_DIR, exist_ok=True)


class PageDimensions(TypedDict):
    width: int
    height: int
    deviceScaleFactor: float


async def fetch_page_content_async(browser_page, wait_for_css: Optional[List[str]]) -> str:
    if wait_for_css:
        for css_selector in wait_for_css:
            await browser_page.wait_for_selector(css_selector)

    await browser_page.screenshot(path=f'{GENERATED_DIR}/example.png')

    dimensions: PageDimensions = await browser_page.evaluate('''() => ({
        width: document.documentElement.clientWidth,
        height: document.documentElement.clientHeight,
        deviceScaleFactor: window.devicePixelRatio
    })''')
    print("Page Dimensions:", dimensions)

    subtitle: str = await browser_page.evaluate('''() => {
        const element = document.querySelector('.results-context-header__job-count');
        return element ? element.innerText.trim() : 'Not found';
    }''')
    print("Job Count:", subtitle)

    return await browser_page.content()


def fetch_page_content_sync(browser_page, wait_for_css: Optional[List[str]]) -> str:
    if wait_for_css:
        for css_selector in wait_for_css:
            browser_page.wait_for_selector(css_selector)

    browser_page.screenshot(path=f'{GENERATED_DIR}/example.png')

    dimensions: PageDimensions = browser_page.evaluate('''() => ({
        width: document.documentElement.clientWidth,
        height: document.documentElement.clientHeight,
        deviceScaleFactor: window.devicePixelRatio
    })''')
    print("Page Dimensions:", dimensions)

    subtitle: str = browser_page.evaluate('''() => {
        const element = document.querySelector('.results-context-header__job-count');
        return element ? element.innerText.trim() : 'Not found';
    }''')
    print("Job Count:", subtitle)

    return browser_page.content()


def scrape_sync(url: str, wait_for_css: Optional[List[str]] = None) -> str:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        browser_page = browser.new_page()
        browser_page.goto(url)
        html_content = fetch_page_content_sync(browser_page, wait_for_css)
        sleep_countdown(3)
        browser.close()
        return html_content


async def scrape_async(url: str, wait_for_css: Optional[List[str]] = None) -> str:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        browser_page = await browser.new_page()
        await browser_page.goto(url)
        html_content = await fetch_page_content_async(browser_page, wait_for_css)
        sleep_countdown(3)
        await browser.close()
        return html_content
