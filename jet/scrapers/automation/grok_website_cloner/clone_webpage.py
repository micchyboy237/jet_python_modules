import os
from pathlib import Path
from typing import List, Literal, TypedDict
from urllib.parse import urljoin, urlparse
import aiofiles
from playwright.async_api import async_playwright, Page
from fake_useragent import UserAgent

from jet.logger import logger


class Resource(TypedDict):
    url: str
    content: bytes
    relative_path: str


async def download_resource(page: Page, resource_url: str, output_dir: str) -> Resource:
    """Download a resource and save it to the output directory."""
    logger.debug(f"Attempting to download resource: {resource_url}")
    try:
        response = await page.context.request.get(resource_url, timeout=30000)
        content = await response.body()
        logger.debug(
            f"Downloaded {resource_url}, content length: {len(content)} bytes")
        parsed_url = urlparse(resource_url)
        filename = os.path.basename(parsed_url.path) or "resource"
        resource_path = os.path.join("assets", filename)
        output_path = os.path.join(output_dir, resource_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        logger.debug(f"Saving resource to: {output_path}")
        async with aiofiles.open(output_path, "wb") as f:
            await f.write(content)
        logger.debug(f"Resource saved successfully: {resource_path}")

        return {"url": resource_url, "content": content, "relative_path": resource_path}
    except Exception as e:
        logger.error(f"Failed to download {resource_url}: {e}")
        return {"url": resource_url, "content": b"", "relative_path": ""}


async def clone_after_render(
    url: str,
    output_dir: str,
    headless: bool = True,
    timeout: int = 10000,
    user_agent_type: Literal["web", "mobile", "random"] = "web"
) -> None:
    """Clone a webpage's HTML and its resources after rendering."""
    logger.debug(
        f"Starting clone for URL: {url}, output_dir: {output_dir}, user_agent_type: {user_agent_type}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        browser = None
        try:
            browser = await p.chromium.launch(headless=headless)
            # Set random user agent based on type
            ua = UserAgent()
            if user_agent_type == "web":
                random_user_agent = ua.chrome  # Desktop Chrome user agent
            elif user_agent_type == "mobile":
                random_user_agent = ua.random  # Fallback to random, filtered for mobile
                while "Mobile" not in random_user_agent and "Android" not in random_user_agent and "iPhone" not in random_user_agent:
                    random_user_agent = ua.random
            else:  # random
                random_user_agent = ua.random
            logger.debug(f"Using user agent: {random_user_agent}")
            context = await browser.new_context(user_agent=random_user_agent)
            page = await context.new_page()

            logger.debug(f"Navigating to {url}")
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=timeout)
                logger.debug("Navigation completed")
            except Exception as e:
                logger.error(f"Navigation failed: {e}")
                raise

            html = await page.content()
            logger.debug(f"HTML content length: {len(html)} characters")

            # Extract resources (CSS, images, etc.)
            resources: List[Resource] = []
            css_links = await page.query_selector_all('link[rel="stylesheet"]')
            logger.debug(f"Found {len(css_links)} CSS links")
            for link in css_links:
                href = await link.get_attribute("href")
                logger.debug(f"Processing CSS link: {href}")
                if href:
                    absolute_url = urljoin(url, href)
                    resource = await download_resource(page, absolute_url, output_dir)
                    resources.append(resource)
                    logger.debug(
                        f"Updating HTML, replacing {href} with {resource['relative_path']}")
                    html = html.replace(href, resource["relative_path"])

            # Save HTML
            output_path = os.path.join(output_dir, "index.html")
            logger.debug(f"Saving HTML to: {output_path}")
            async with aiofiles.open(output_path, "w", encoding="utf-8") as f:
                await f.write(html)
            logger.debug(f"HTML saved successfully")
        except Exception as e:
            logger.error(f"Error in clone_after_render: {e}")
            raise
        finally:
            if browser:
                await browser.close()
                logger.debug("Browser closed")
