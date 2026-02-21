import asyncio

from crawl4ai import AsyncWebCrawler, BrowserConfig


async def main():
    browser_config = BrowserConfig(
        headless=True,
        enable_stealth=True,  # ← playwright-stealth
        user_agent_mode="random",
        viewport_width=1366 + 20,  # slight randomness helps
        viewport_height=768 + 30,
        verbose=False,
    )

    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun("https://bot.sannysoft.com")
        print("Fingerprint result:", result.markdown[:300])


if __name__ == "__main__":
    asyncio.run(main())
