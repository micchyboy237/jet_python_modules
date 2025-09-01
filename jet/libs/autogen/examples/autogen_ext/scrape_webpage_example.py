import os
import shutil
import asyncio
import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from playwright.async_api import async_playwright, Page
from jet.libs.autogen.playwright_controller import PlaywrightController
from jet.logger import CustomLogger

# Define output directory for logs, scraped data, and screenshots
OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(
        os.path.basename(__file__))[0]
)
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.orange(f"Logs: {log_file}")


async def scrape_webpage_example() -> None:
    """
    Demonstrates using PlaywrightController to visit a URL, scrape title, metadata, body text, and links,
    save results to a JSON file, and capture screenshots in the output directory.
    """
    # Define screenshot directory and timestamp for unique filenames
    screenshot_dir = f"{OUTPUT_DIR}/screenshots"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(screenshot_dir, exist_ok=True)

    # Define URL to scrape
    target_url = "https://www.wikipedia.org"

    async with async_playwright() as p:
        # Initialize browser and page
        browser = await p.chromium.launch(headless=False)
        try:
            context = await browser.new_context()
            page = await context.new_page()

            # Visit the URL and wait for content to load
            controller = PlaywrightController(default_zoom=0.5)
            reset_metadata, reset_download = await controller.visit_page(page, target_url)
            await page.wait_for_load_state("networkidle")
            logger.info(
                f"Visited URL: {target_url}, reset_metadata: {reset_metadata}, reset_download: {reset_download}")

            # Save screenshot of initial page
            initial_screenshot_path = os.path.join(
                screenshot_dir, f"initial_page_{timestamp}.png")
            await page.screenshot(path=initial_screenshot_path)
            logger.info(
                f"Saved initial page screenshot to: {initial_screenshot_path}")

            # Scrape title
            page_title = await page.title()
            logger.info(f"Scraped page title: {page_title}")

            # Scrape metadata
            metadata: Dict[str, Any] = await controller.get_page_metadata(page)
            logger.info(f"Scraped metadata: {json.dumps(metadata, indent=2)}")

            # Scrape visible body text
            body_text = await controller.get_visible_text(page)
            logger.info(
                f"Scraped body text (first 100 chars): {body_text[:100] + '...' if len(body_text) > 100 else body_text}")

            # Scrape visible links
            links: List[Dict[str, str]] = await controller.get_links(page, base_url=target_url)
            logger.info(
                f"Scraped {len(links)} links (first 2): {json.dumps(links[:2], indent=2)}")

            # Save scraped data to JSON
            scraped_data = {
                "url": target_url,
                "title": page_title,
                "metadata": metadata,
                "body_text": body_text,
                "links": links
            }
            output_file = os.path.join(
                OUTPUT_DIR, f"scraped_data_{timestamp}.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(scraped_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved scraped data to: {output_file}")

            # Save screenshot after scraping
            scraped_screenshot_path = os.path.join(
                screenshot_dir, f"scraped_page_{timestamp}.png")
            await page.screenshot(path=scraped_screenshot_path)
            logger.info(
                f"Saved scraped page screenshot to: {scraped_screenshot_path}")

        except Exception as e:
            logger.error(f"Error during scraping: {str(e)}")
            raise

        finally:
            await browser.close()

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    asyncio.run(scrape_webpage_example())
