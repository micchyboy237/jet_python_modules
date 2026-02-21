import asyncio
import os

from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig

ROOT_URL = os.getenv("SEARXNG_URL")


async def main():
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,  # quiet in production
        viewport_width=1280,
        viewport_height=800,
    )

    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(
            url=ROOT_URL,
            config=CrawlerRunConfig(cache_mode=CacheMode.BYPASS),
        )
        print(result.markdown[:400])


if __name__ == "__main__":
    asyncio.run(main())
