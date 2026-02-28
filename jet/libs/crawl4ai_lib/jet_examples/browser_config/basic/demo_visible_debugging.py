import asyncio
import os

from crawl4ai import AsyncWebCrawler, BrowserConfig

ROOT_URL = os.getenv("SEARXNG_URL")


async def main():
    browser_config = BrowserConfig(
        headless=False,  # ← see what's happening
        verbose=True,
        viewport_width=1440,
        viewport_height=900,
    )

    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(ROOT_URL)
        print("Success:", result.success)
        print("Status:", result.status_code)


if __name__ == "__main__":
    asyncio.run(main())
