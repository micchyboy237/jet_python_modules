import time
from fake_useragent import UserAgent
import asyncio
from jet.cache.redis.types import RedisConfigParams
from jet.cache.redis.utils import RedisCache
from jet.logger import logger
from jet.scrapers.utils import validate_headers
from playwright.async_api import async_playwright
from typing import List

REDIS_CONFIG = RedisConfigParams(
    port=3102
)


async def fetch_page_content(page, url: str) -> str:
    try:
        await page.goto(url)
        await page.wait_for_load_state("load", timeout=10000)  # 10 seconds
        content = await page.content()
        return content
    except Exception as e:
        logger.warning(f"Failed to load {url}: {e}")
        return ""


async def scrape_with_playwright(urls: List[str]) -> List[str]:
    ua = UserAgent()
    user_agent = ua.chrome  # Or ua.random for truly random

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                '--window-size=1512,982',  # MacBook M1 default viewport size
                '--force-device-scale-factor=0.8',  # Set zoom to 80%
                "--disable-infobars",
                "--disable-notifications",
                "--disable-gpu",
                "--disable-extensions",
                "--disable-dev-shm-usage",
                "--no-sandbox",  # For Linux server
                "--incognito",
                "--mute-audio"
            ]
        )
        context = await browser.new_context(
            user_agent=user_agent,
            color_scheme='dark',
        )

        pages = [await context.new_page() for _ in urls]
        tasks = [fetch_page_content(page, url)
                 for page, url in zip(pages, urls)]
        html_results = await asyncio.gather(*tasks)

        await browser.close()
        return html_results


async def scrape_multiple_urls(urls: List[str], top_n: int = 3, num_parallel: int = 3) -> List[str]:
    cache = RedisCache(config=REDIS_CONFIG)
    # Initialize result list to maintain order
    html_results = [None] * len(urls)
    uncached_urls = []
    uncached_indices = []

    # Step 1: Check cache for each URL
    for i, url in enumerate(urls):
        cache_key = f"html:{url}"  # Unique cache key for each URL
        cached_content = cache.get(cache_key)

        if cached_content:
            # Cache hit: Store result directly
            logger.success(f"Cache hit for {url}")
            html_results[i] = cached_content['content']
        else:
            # Cache miss: Mark for scraping
            logger.warning(f"Cache miss for {url}, will scrape...")
            uncached_urls.append(url)
            uncached_indices.append(i)

    # Step 2: Scrape uncached URLs in batches
    if uncached_urls:
        logger.info(
            f"Scraping {len(uncached_urls)} uncached URLs in batches of {num_parallel}...")
        for i in range(0, len(uncached_urls), num_parallel):
            batch_urls = uncached_urls[i:i + num_parallel]
            batch_indices = uncached_indices[i:i + num_parallel]
            scraped_results = await scrape_with_playwright(batch_urls)

            # Step 3: Store scraped results in cache and assign to results
            results_count = 0
            for idx, html_content, url in zip(batch_indices, scraped_results, batch_urls):
                if html_content and validate_headers(html_content, min_count=5):
                    # Store in cache with TTL of 3600 seconds (1 hour)
                    cache_key = f"html:{url}"
                    cache.set(cache_key, {'content': html_content}, ttl=3600)
                    html_results[idx] = html_content
                    results_count += 1

                    if results_count == top_n:
                        return html_results
                else:
                    continue

    return html_results

if __name__ == "__main__":
    urls = [
        "https://example.com",
        "https://httpbin.org/html",
        "https://www.python.org",
        "https://www.reddit.com/r/nba/",
    ]
    # Run the async function
    html_list = asyncio.run(scrape_multiple_urls(urls))
    for i, html in enumerate(html_list):
        print(f"--- HTML from {urls[i]} ---\n{html[:300]}...\n")
