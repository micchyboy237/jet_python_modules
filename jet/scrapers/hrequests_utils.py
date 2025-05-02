import aiohttp
import asyncio
from fake_useragent import UserAgent
from typing import List, Optional
from jet.code.splitter_markdown_utils import get_md_header_contents
from jet.logger import logger
from jet.scrapers.utils import scrape_links
from jet.cache.redis.types import RedisConfigParams
from jet.cache.redis.utils import RedisCache

REDIS_CONFIG = RedisConfigParams(
    port=3102
)
cache = RedisCache(config=REDIS_CONFIG)


async def scrape_url(session: aiohttp.ClientSession, url: str, ua: UserAgent) -> Optional[str]:
    cache_key = f"html:{url}"
    cached_content = cache.get(cache_key)

    if cached_content:
        return cached_content['content']

    logger.warning(f"Cache miss for {url}")

    try:
        headers = {'User-Agent': ua.random}
        async with session.get(url, headers=headers) as response:
            if response.status == 200:
                logger.success(f"Scraped {url}")
                html_content = await response.text()
                cache.set(cache_key, {'content': html_content}, ttl=3600)
                return html_content
            else:
                logger.warning(
                    f"Failed: {url} - Status Code: {response.status}, Reason: {response.reason}")
                return None
    except Exception as e:
        logger.error(f"Error fetching {url}: {str(e)}")
        return None


async def scrape_urls(urls: List[str], num_parallel: int = 5) -> List[Optional[str]]:
    ua = UserAgent()
    semaphore = asyncio.Semaphore(num_parallel)

    async def sem_fetch(url: str, session: aiohttp.ClientSession) -> Optional[str]:
        async with semaphore:
            return await scrape_url(session, url, ua)

    async with aiohttp.ClientSession() as session:
        tasks = [sem_fetch(url, session) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        final_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Task failed with exception: {str(result)}")
                final_results.append(None)
            else:
                final_results.append(result)
        return final_results

if __name__ == "__main__":
    urls = [
        "https://www.imdb.com/list/ls505070747",
        "https://myanimelist.net/stacks/32507",
        "https://example.com",
        "https://python.org",
        "https://github.com",
        "https://httpbin.org/html",
        "https://www.wikipedia.org/",
        "https://www.mozilla.org",
        "https://www.stackoverflow.com"
    ]
    results = asyncio.run(scrape_urls(urls, num_parallel=3))
    for url, html_str in zip(urls, results):
        if html_str:
            all_links = scrape_links(html_str, base_url=url)
            headers = get_md_header_contents(html_str)
            logger.success(
                f"Scraped {url}, headers length: {len(headers)}, links count: {len(all_links)}")
        else:
            logger.error(f"Failed to fetch {url}")
