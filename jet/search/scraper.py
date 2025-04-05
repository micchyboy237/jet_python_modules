from jet.scrapers.browser.playwright import scrape_sync, setup_browser_page, setup_sync_browser_page
from jet.scrapers.browser.selenium_utils import UrlScraper
from jet.scrapers.preprocessor import scrape_markdown
import hashlib
import os
import json
from jet.cache.redis import RedisCache, RedisConfigParams
from jet.logger import logger
from jet.scrapers.hrequests import request_url

REDIS_CONFIG = RedisConfigParams(
    port=3102
)

# cache_dir = os.path.join(os.getcwd(), "cache")
# os.makedirs(cache_dir, exist_ok=True)


# def cache_file_path(url: str) -> str:
#     url_hash = hashlib.md5(url.encode("utf-8")).hexdigest()
#     return os.path.join(cache_dir, f"{url_hash}.json")


# def load_cache(url: str) -> dict:
#     cache_path = cache_file_path(url)
#     if os.path.exists(cache_path):
#         with open(cache_path, "r", encoding="utf-8") as f:
#             return json.load(f)
#     return None


# def save_cache(url: str, data: dict):
#     cache_path = cache_file_path(url)
#     with open(cache_path, "w", encoding="utf-8") as f:
#         json.dump(data, f, ensure_ascii=False, indent=4)

def main_selenium(url):
    url_scraper = UrlScraper()
    html_str = url_scraper.scrape_url(url)
    return html_str


def main_hrequests(url):
    html_parser = request_url(url)
    html_str = html_parser.raw_html.decode('utf-8')
    return html_str


def scrape_url(url: str, config: RedisConfigParams = REDIS_CONFIG, show_browser: bool = False) -> str:
    cache = RedisCache(config=config)
    cache_key = url
    cached_result = cache.get(cache_key)

    if cached_result:
        logger.log(f"scrape_url: Cache hit for", cache_key,
                   colors=["LOG", "BRIGHT_SUCCESS"])
        return cached_result

    logger.info(f"scrape_url: Cache miss for {cache_key}")

    if show_browser:
        # html_str = main_selenium(url)
        html_str = scrape_sync(url)["html"]
    else:
        html_str = main_hrequests(url)

    cache.set(cache_key, html_str)
    return html_str
