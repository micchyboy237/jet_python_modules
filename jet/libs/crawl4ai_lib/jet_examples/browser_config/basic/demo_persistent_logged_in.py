import asyncio
import os

from crawl4ai import AsyncWebCrawler, BrowserConfig


async def main():
    user_data_dir = os.path.expanduser("~/my-crawl4ai-profile")

    browser_config = BrowserConfig(
        headless=False,  # first time — login manually
        use_persistent_context=True,
        user_data_dir=user_data_dir,
        verbose=True,
    )

    async with AsyncWebCrawler(config=browser_config) as crawler:
        # First run: you can manually login in the visible browser
        # Then close it → subsequent runs will reuse cookies/session
        result = await crawler.arun("https://example.com/dashboard")  # protected page
        print("Logged in content preview:", result.markdown[:250])


if __name__ == "__main__":
    asyncio.run(main())
