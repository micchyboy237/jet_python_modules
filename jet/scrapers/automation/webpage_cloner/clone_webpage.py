import asyncio
import os
from pathlib import Path
import random
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
        response = await page.context.request.get(resource_url, timeout=10000)
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
    user_agent_type: Literal["web", "mobile", "random"] = "web",
    max_retries: int = 3
) -> None:
    """Clone a webpage's HTML and its resources after rendering."""
    logger.debug(
        f"Starting clone for URL: {url}, output_dir: {output_dir}, user_agent_type: {user_agent_type}, max_retries: {max_retries}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        browser = None
        context = None
        page = None
        try:
            browser = await p.chromium.launch(headless=headless)
            ua = UserAgent()

            for attempt in range(max_retries + 1):
                # Add random delay before retry (skip for first attempt)
                if attempt > 0:
                    # Random delay between 1 and 5 seconds
                    delay = random.uniform(1, 5)
                    logger.debug(
                        f"Waiting for {delay:.2f} seconds before retry attempt {attempt + 1}")
                    await asyncio.sleep(delay)

                # Create new context for each attempt
                if context:
                    await context.close()
                    logger.debug(f"Closed context for attempt {attempt}")

                # Set random user agent based on type
                if user_agent_type == "web":
                    random_user_agent = ua.chrome
                elif user_agent_type == "mobile":
                    random_user_agent = ua.random
                    while "Mobile" not in random_user_agent and "Android" not in random_user_agent and "iPhone" not in random_user_agent:
                        random_user_agent = ua.random
                else:  # random
                    random_user_agent = ua.random
                logger.debug(
                    f"Attempt {attempt + 1}/{max_retries + 1} with user agent: {random_user_agent}")

                context = await browser.new_context(user_agent=random_user_agent)
                page = await context.new_page()

                logger.debug(f"Navigating to {url} on attempt {attempt + 1}")
                try:
                    await page.goto(url, wait_until="domcontentloaded", timeout=timeout)
                    logger.debug("Navigation completed")
                    break  # Success, exit retry loop
                except Exception as e:
                    logger.warning(
                        f"Navigation failed on attempt {attempt + 1}: {e}")
                    if attempt == max_retries:
                        logger.error(f"All {max_retries + 1} attempts failed")
                        raise Exception(
                            f"Failed to navigate to {url} after {max_retries + 1} attempts")

            # Extract HTML after successful navigation
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
            if context:
                await context.close()
                logger.debug("Context closed")
            if browser:
                await browser.close()
                logger.debug("Browser closed")
