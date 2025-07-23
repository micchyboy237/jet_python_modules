import asyncio
import os
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

from jet.file.utils import save_file


async def clone_after_render(url: str, out_folder: str = 'playwright_mirror') -> None:
    os.makedirs(out_folder, exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(url, wait_until="networkidle")
        html = await page.content()
        await browser.close()

    soup = BeautifulSoup(html, 'html.parser')

    # Similar rewrite_links usage here...
    save_file(soup.prettify(), os.path.join(out_folder, 'index.html'))


def run_clone_after_render(url: str, out_folder: str = 'playwright_mirror') -> None:
    asyncio.run(clone_after_render(url, out_folder))
