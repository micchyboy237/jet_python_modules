import time
from fake_useragent import UserAgent
import asyncio
from jet.cache.redis.types import RedisConfigParams
from jet.cache.redis.utils import RedisCache
from jet.logger import logger
from jet.scrapers.utils import validate_headers
from playwright.async_api import async_playwright
from typing import Any, AsyncGenerator, List, Tuple

REDIS_CONFIG = RedisConfigParams(
    port=3102
)


async def fetch_page_content(page, url: str) -> Tuple[int, str, str]:
    try:
        logger.debug(f"Scraping url: {url}")
        await page.goto(url)
        await page.wait_for_load_state("load", timeout=10000)  # 10 seconds
        content = await page.content()
        logger.success(f"Done scraping url: {url}")
        return (0, url, content)
    except Exception as e:
        logger.warning(f"Failed to load {url}: {e}")
        return (0, url, "")


async def scrape_with_playwright(urls: List[str]) -> List[str]:
    ua = UserAgent()
    user_agent = ua.chrome

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                '--window-size=1512,982',
                '--force-device-scale-factor=0.8',
                "--disable-infobars",
                "--disable-notifications",
                "--disable-gpu",
                "--disable-extensions",
                "--disable-dev-shm-usage",
                "--no-sandbox",
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


async def scrape_multiple_urls(urls: List[str], top_n: int = 3, num_parallel: int = 3, max_retries: int = 1) -> AsyncGenerator[Tuple[str, Any], None]:
    cache = RedisCache(config=REDIS_CONFIG)
    html_results = [None] * len(urls)
    uncached_urls = []
    uncached_indices = []
    retries = {url: 0 for url in urls}  # Track retries per URL

    # Check cache for each URL
    for i, url in enumerate(urls):
        cache_key = f"html:{url}"
        cached_content = cache.get(cache_key)
        if cached_content:
            logger.success(f"Cache hit for {url}")
            html_results[i] = cached_content['content']
            if cached_content['content'] and validate_headers(cached_content['content'], min_count=5):
                yield url, cached_content['content']
        else:
            logger.warning(f"Cache miss for {url}, will scrape...")
            uncached_urls.append(url)
            uncached_indices.append(i)

    # Scrape uncached URLs dynamically
    if uncached_urls:
        logger.info(
            f"Scraping {len(uncached_urls)} uncached URLs with max {num_parallel} parallel tasks...")
        active_tasks = []
        results_count = 0
        url_queue = list(zip(uncached_indices, uncached_urls)
                         )  # [(index, url), ...]
        in_progress_urls = set()

        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    '--window-size=1512,982',
                    '--force-device-scale-factor=0.8',
                    "--disable-infobars",
                    "--disable-notifications",
                    "--disable-gpu",
                    "--disable-extensions",
                    "--disable-dev-shm-usage",
                    "--no-sandbox",
                    "--incognito",
                    "--mute-audio"
                ]
            )
            try:
                context = await browser.new_context(
                    user_agent=UserAgent().chrome,
                    color_scheme='dark',
                )

                async def process_url(index: int, url: str):
                    page = await context.new_page()
                    try:
                        result = await fetch_page_content(page, url)
                        return (index, url, result[2])
                    finally:
                        await page.close()

                # Start initial tasks up to num_parallel
                while url_queue and len(active_tasks) < num_parallel and results_count < top_n:
                    index, url = url_queue.pop(0)
                    in_progress_urls.add(url)
                    task = asyncio.create_task(process_url(index, url))
                    active_tasks.append((task, index, url))

                # Process tasks and add new ones dynamically
                while active_tasks and results_count < top_n:
                    done, _ = await asyncio.wait(
                        [task for task, _, _ in active_tasks],
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    for completed_task in done:
                        try:
                            index, url, html_content = completed_task.result()
                        except Exception as e:
                            logger.error(
                                f"Task for {url} failed with error: {e}")
                            index, url = next(
                                (i, u) for t, i, u in active_tasks if t == completed_task)
                            html_content = ""

                        in_progress_urls.discard(url)
                        active_tasks = [
                            (t, i, u) for t, i, u in active_tasks if t != completed_task]

                        if html_content and validate_headers(html_content, min_count=5):
                            logger.success(f"Valid content scraped for {url}")
                            cache_key = f"html:{url}"
                            cache.set(
                                cache_key, {'content': html_content}, ttl=3600)
                            html_results[index] = html_content
                            results_count += 1
                            logger.info(
                                f"Results count: {results_count} / {top_n}")
                            yield url, html_content
                        else:
                            logger.warning(
                                f"Invalid or empty content for {url}")
                            if retries[url] < max_retries:
                                retries[url] += 1
                                logger.info(
                                    f"Retrying {url} (attempt {retries[url]}/{max_retries})")
                                # Requeue for retry
                                url_queue.append((index, url))
                            else:
                                logger.error(
                                    f"Max retries reached for {url}, skipping")

                        # Add new tasks if possible
                        while url_queue and len(active_tasks) < num_parallel and results_count < top_n:
                            new_index, new_url = url_queue.pop(0)
                            if new_url not in in_progress_urls:
                                in_progress_urls.add(new_url)
                                new_task = asyncio.create_task(
                                    process_url(new_index, new_url))
                                active_tasks.append(
                                    (new_task, new_index, new_url))

            finally:
                await browser.close()

    logger.info(f"Scraping completed with {results_count} valid results")

if __name__ == "__main__":
    import sys

    async def main():
        sample_urls = [
            "https://example.com",
            "https://httpbin.org/html",
            "https://www.wikipedia.org/",
            "https://www.bbc.com/news",
            "https://www.cnn.com",
            "https://www.nytimes.com",
            "https://www.mozilla.org",
            "https://www.stackoverflow.com",
            "https://news.ycombinator.com",
            "https://www.reddit.com"
        ]

        print("Starting scrape...")
        async for status, content in scrape_multiple_urls(sample_urls, top_n=5, num_parallel=3):
            if status == "in_progress":
                logger.info(f"Scraping in progress for: {content}")
            else:
                logger.success(
                    f"Scraped {status}, content length: {len(content)}")

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.warning("Scraping interrupted by user.")
        sys.exit(0)
