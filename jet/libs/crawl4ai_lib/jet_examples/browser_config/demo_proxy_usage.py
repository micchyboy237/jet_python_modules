import asyncio

from crawl4ai import AsyncWebCrawler, BrowserConfig


async def main():
    browser_config = BrowserConfig(
        headless=True,
        proxy_config={
            "server": "http://proxy.example.com:8080",
            "username": "user123",
            "password": "pass456",
        },
        verbose=True,
    )

    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun("https://api.ipify.org?format=json")
        print("IP seen by target:", result.markdown.strip())


if __name__ == "__main__":
    asyncio.run(main())
