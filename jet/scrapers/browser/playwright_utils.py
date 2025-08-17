import time
from fake_useragent import UserAgent
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from jet.cache.redis.types import RedisConfigParams
from jet.cache.redis.utils import RedisCache
from jet.code.splitter_markdown_utils import get_md_header_contents
from jet.logger import logger
from jet.scrapers.utils import extract_title_and_metadata, scrape_links, validate_headers
from playwright.async_api import async_playwright
from playwright.sync_api import sync_playwright
from typing import Any, AsyncGenerator, List, Tuple, Generator

REDIS_CONFIG = RedisConfigParams(
    port=6379
)


async def fetch_page_content(page, url: str) -> Tuple[int, str, str, List[str]]:
    try:
        logger.debug(f"Scraping url: {url}")
        await page.goto(url, timeout=5000)
        await page.wait_for_load_state("load", timeout=5000)
        html_str = await page.content()
        all_links = scrape_links(html_str, base_url=url)
        # title_and_metadata = extract_title_and_metadata(html_str)
        logger.success(f"Done scraping url: {url} | Links ({len(all_links)})")
        return (0, url, html_str, all_links)
    except Exception as e:
        logger.warning(f"Failed to load {url}: {e}")
        return (0, url, "", [])


def sync_fetch_page_content(page, url: str) -> Tuple[int, str, str, List[str]]:
    try:
        logger.debug(f"Scraping url: {url}")
        page.goto(url, timeout=5000)
        page.wait_for_load_state("load", timeout=5000)
        html_str = page.content()
        all_links = scrape_links(html_str, base_url=url)
        # title_and_metadata = extract_title_and_metadata(html_str)
        logger.success(f"Done scraping url: {url} | Links ({len(all_links)})")
        return (0, url, html_str, all_links)
    except Exception as e:
        logger.warning(f"Failed to load {url}: {e}")
        return (0, url, "", [])


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


def sync_scrape_with_playwright(urls: List[str]) -> List[str]:
    ua = UserAgent()
    user_agent = ua.chrome

    with sync_playwright() as p:
        browser = p.chromium.launch(
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
        context = browser.new_context(
            user_agent=user_agent,
            color_scheme='dark',
        )

        results = []
        for url in urls:
            page = context.new_page()
            result = sync_fetch_page_content(page, url)
            results.append(result)
            page.close()

        browser.close()
        return results


async def ascrape_multiple_urls(urls: List[str], top_n: int = 3, num_parallel: int = 3, min_header_count: int = 5, min_avg_word_count: int = 20, max_retries: int = 1) -> AsyncGenerator[Tuple[str, Any], None]:
    cache = RedisCache(config=REDIS_CONFIG)
    html_results = [None] * len(urls)  # For original URLs
    uncached_urls = []
    uncached_indices = []
    retries = {url: 0 for url in urls}
    results_count = 0
    # Track processed or queued URLs to avoid duplicates
    processed_urls = set(urls)
    # (index, url) for original URLs
    url_queue = list(zip(range(len(urls)), urls))

    # Check cache for each original URL
    for i, url in enumerate(urls):
        cache_key = f"html:{url}"
        cached_content = cache.get(cache_key)
        if cached_content:
            logger.success(f"Cache hit for {url}")
            html_results[i] = cached_content['content']
            if cached_content['content'] and validate_headers(cached_content['content'], min_count=min_header_count, min_avg_word_count=min_avg_word_count) and results_count < top_n:
                results_count += 1
                logger.info(
                    f"Yielding cached result for {url} ({results_count}/{top_n})")
                yield url, cached_content['content']
                if results_count >= top_n:
                    logger.info("Reached top_n valid results, stopping")
                    return
        else:
            # logger.warning(f"Cache miss for {url}, will scrape...")
            uncached_urls.append(url)
            uncached_indices.append(i)

    if results_count >= top_n:
        logger.info("Reached top_n valid results from cache, stopping")
        return

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
                    # Include all_links
                    return (index, url, result[2], result[3])
                finally:
                    await page.close()

            in_progress_urls = set()
            while url_queue and results_count < top_n:
                active_tasks = []
                while url_queue and len(active_tasks) < num_parallel and results_count < top_n:
                    index, url = url_queue.pop(0)
                    in_progress_urls.add(url)
                    task = asyncio.create_task(process_url(index, url))
                    active_tasks.append((task, index, url))

                if not active_tasks:
                    break

                done, _ = await asyncio.wait(
                    [task for task, _, _ in active_tasks],
                    return_when=asyncio.FIRST_COMPLETED
                )

                for completed_task in done:
                    index, url = next(
                        (i, u) for t, i, u in active_tasks if t == completed_task)
                    active_tasks = [
                        (t, i, u) for t, i, u in active_tasks if t != completed_task]
                    in_progress_urls.discard(url)

                    try:
                        task_index, task_url, html_content, all_links = completed_task.result()
                    except Exception as e:
                        logger.error(f"Task for {url} failed with error: {e}")
                        html_content, all_links = "", []

                    # Process the scraped content
                    if html_content and validate_headers(html_content, min_count=min_header_count, min_avg_word_count=min_avg_word_count) and results_count < top_n:
                        logger.success(f"Valid content scraped for {url}")
                        cache_key = f"html:{url}"
                        cache.set(
                            cache_key, {'content': html_content}, ttl=3600)
                        # Store in html_results for original URLs
                        if task_index < len(html_results):
                            html_results[task_index] = html_content
                        results_count += 1
                        logger.info(
                            f"Yielding scraped result for {url} ({results_count}/{top_n})")
                        yield url, html_content
                    else:
                        if html_content:
                            logger.warning(
                                f"Content for {url} ignored as top_n reached or invalid")
                        else:
                            logger.warning(
                                f"Invalid or empty content for {url}")
                        if url in retries and retries[url] < max_retries:
                            retries[url] += 1
                            logger.info(
                                f"Retrying {url} (attempt {retries[url]}/{max_retries})")
                            url_queue.append((index, url))
                        else:
                            logger.error(
                                f"Max retries reached for {url}, moving to next URL")

                    # Add new links to the queue
                    for link in all_links:
                        if link not in processed_urls and link not in in_progress_urls:
                            processed_urls.add(link)
                            # Initialize retries for new link
                            retries[link] = 0
                            # Use a new index beyond the original urls length
                            new_index = len(html_results) + len(processed_urls)
                            url_queue.append((new_index, link))
                            # logger.debug(f"Added link to queue: {link}")

            # Cancel any remaining tasks
            for task, _, _ in active_tasks:
                task.cancel()
            await asyncio.gather(*[task for task, _, _ in active_tasks], return_exceptions=True)

        finally:
            await browser.close()

    logger.info(f"Scraping completed with {results_count} valid results")


def scrape_multiple_urls(urls: List[str], top_n: int = 3, num_parallel: int = 3, min_header_count: int = 5, min_avg_word_count: int = 20, max_retries: int = 1) -> Generator[Tuple[str, Any], None, None]:
    cache = RedisCache(config=REDIS_CONFIG)
    html_results = [None] * len(urls)
    uncached_urls = []
    uncached_indices = []
    retries = {url: 0 for url in urls}
    results_count = 0
    lock = threading.Lock()
    processed_urls = set(urls)  # Track processed or queued URLs
    # (index, url) for original URLs
    url_queue = list(zip(range(len(urls)), urls))

    # Check cache for each URL
    for i, url in enumerate(urls):
        cache_key = f"html:{url}"
        cached_content = cache.get(cache_key)
        if cached_content:
            logger.success(f"Cache hit for {url}")
            html_results[i] = cached_content['content']
            if cached_content['content'] and validate_headers(cached_content['content'], min_count=min_header_count, min_avg_word_count=min_avg_word_count) and results_count < top_n:
                with lock:
                    results_count += 1
                logger.info(
                    f"Yielding cached result for {url} ({results_count}/{top_n})")
                yield url, cached_content['content']
                if results_count >= top_n:
                    logger.info("Reached top_n valid results, stopping")
                    return
        else:
            # logger.warning(f"Cache miss for {url}, will scrape...")
            uncached_urls.append(url)
            uncached_indices.append(i)

    if results_count >= top_n:
        logger.info("Reached top_n valid results from cache, stopping")
        return

    if uncached_urls:
        logger.info(
            f"Scraping {len(uncached_urls)} uncached URLs with max {num_parallel} parallel threads...")

        def process_url(index: int, url: str) -> Tuple[int, str, str, List[str]]:
            with sync_playwright() as p:
                browser = p.chromium.launch(
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
                    context = browser.new_context(
                        user_agent=UserAgent().chrome,
                        color_scheme='dark',
                    )
                    page = context.new_page()
                    try:
                        result = sync_fetch_page_content(page, url)
                        # Include all_links
                        return (index, url, result[2], result[3])
                    finally:
                        page.close()
                finally:
                    browser.close()

        with ThreadPoolExecutor(max_workers=num_parallel) as executor:
            while url_queue and results_count < top_n:
                batch = url_queue[:num_parallel]
                url_queue = url_queue[num_parallel:]

                futures = []
                future_to_url_index = {}
                for index, url in batch:
                    future = executor.submit(process_url, index, url)
                    futures.append(future)
                    future_to_url_index[future] = (index, url)

                for future in futures:
                    index, url = future_to_url_index[future]
                    try:
                        task_index, task_url, html_content, all_links = future.result()
                    except Exception as e:
                        logger.error(
                            f"Thread for {url} failed with error: {e}")
                        html_content, all_links = "", []

                    with lock:
                        if html_content and validate_headers(html_content, min_count=min_header_count, min_avg_word_count=min_avg_word_count) and results_count < top_n:
                            logger.success(f"Valid content scraped for {url}")
                            cache_key = f"html:{url}"
                            cache.set(
                                cache_key, {'content': html_content}, ttl=3600)
                            if task_index < len(html_results):
                                html_results[task_index] = html_content
                            results_count += 1
                            logger.info(
                                f"Yielding scraped result for {url} ({results_count}/{top_n})")
                            yield url, html_content
                        else:
                            if html_content:
                                logger.warning(
                                    f"Content for {url} ignored as top_n reached or invalid")
                            else:
                                logger.warning(
                                    f"Invalid or empty content for {url}")
                            if url in retries and retries[url] < max_retries:
                                retries[url] += 1
                                logger.info(
                                    f"Retrying {url} (attempt {retries[url]}/{max_retries})")
                                url_queue.append((index, url))
                            else:
                                logger.error(
                                    f"Max retries reached for {url}, moving to next URL")

                        # Add new links to the queue
                        for link in all_links:
                            if link not in processed_urls:
                                processed_urls.add(link)
                                retries[link] = 0
                                new_index = len(html_results) + \
                                    len(processed_urls)
                                url_queue.append((new_index, link))
                                # logger.debug(f"Added link to queue: {link}")

                    if results_count >= top_n:
                        url_queue.clear()
                        break

    logger.info(f"Scraping completed with {results_count} valid results")


if __name__ == "__main__":
    import sys

    async def main():
        sample_urls = [
            "https://www.imdb.com/list/ls505070747",
            "https://myanimelist.net/stacks/32507",
            "https://example.com",
            "https://httpbin.org/html",
            "https://www.wikipedia.org/",
            "https://www.bbc.com/news",
            "https://www.cnn.com",
            "https://www.nytimes.com",
            "https://www.mozilla.org",
            "https://www.stackoverflow.com"
        ]

        print("\nStarting sync scrape...")
        for url, html_str in scrape_multiple_urls(sample_urls, top_n=5, num_parallel=3, max_retries=1):
            all_links = scrape_links(html_str, base_url=url)
            headers = get_md_header_contents(html_str)
            logger.success(
                f"Scraped {url}, headers length: {len(headers)}, links count: {len(all_links)}")

        print("\nStarting async scrape...")
        async for url, html_str in ascrape_multiple_urls(sample_urls, top_n=5, num_parallel=3, max_retries=1):
            all_links = scrape_links(html_str, base_url=url)
            headers = get_md_header_contents(html_str)
            logger.success(
                f"Scraped {url}, headers length: {len(headers)}, links count: {len(all_links)}")

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.warning("Scraping interrupted by user.")
        sys.exit(0)
