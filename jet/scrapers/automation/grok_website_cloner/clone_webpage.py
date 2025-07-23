import os
from pathlib import Path
from typing import List, TypedDict
from urllib.parse import urljoin, urlparse
import aiofiles
from playwright.async_api import async_playwright, Page

from jet.logger import logger


class Resource(TypedDict):
    url: str
    content: bytes
    relative_path: str


async def download_resource(page: Page, resource_url: str, output_dir: str) -> Resource:
    """Download a resource and save it to the output directory."""
    try:
        response = await page.context.request.get(resource_url)
        content = await response.body()
        parsed_url = urlparse(resource_url)
        filename = os.path.basename(parsed_url.path) or "resource"
        resource_path = os.path.join("assets", filename)
        output_path = os.path.join(output_dir, resource_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        async with aiofiles.open(output_path, "wb") as f:
            await f.write(content)

        return {"url": resource_url, "content": content, "relative_path": resource_path}
    except Exception as e:
        print(f"Failed to download {resource_url}: {e}")
        return {"url": resource_url, "content": b"", "relative_path": ""}


async def clone_after_render(url: str, output_dir: str) -> None:
    """Clone a webpage's HTML and its resources after rendering."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()

        await page.goto(url, wait_until="networkidle")
        html = await page.content()

        # Extract resources (CSS, images, etc.)
        resources: List[Resource] = []
        css_links = await page.query_selector_all('link[rel="stylesheet"]')
        for link in css_links:
            href = await link.get_attribute("href")
            if href:
                absolute_url = urljoin(url, href)
                resource = await download_resource(page, absolute_url, output_dir)
                resources.append(resource)
                # Update HTML to point to local resource
                html = html.replace(href, resource["relative_path"])

        # Save HTML
        output_path = os.path.join(output_dir, "index.html")
        async with aiofiles.open(output_path, "w", encoding="utf-8") as f:
            await f.write(html)
            logger.success(f"Saved HTML to {output_path}")

        await browser.close()
