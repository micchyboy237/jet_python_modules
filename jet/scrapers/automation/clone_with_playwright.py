import os
from pathlib import Path
from playwright.async_api import async_playwright


async def clone_after_render(url: str, output_dir: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()

        await page.goto(url, wait_until='networkidle')
        html = await page.content()

        output_path = os.path.join(output_dir, "index.html")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        await browser.close()
