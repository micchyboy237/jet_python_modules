from jet.scrapers.selenium import UrlScraper
from jet.scrapers.preprocessor import scrape_markdown
import hashlib
import os
import json
from .cache import Cache
from jet.cache.redis import RedisConfigParams
from jet.logger import logger

cache_dir = os.path.join(os.getcwd(), "cache")
os.makedirs(cache_dir, exist_ok=True)


def cache_file_path(url: str) -> str:
    url_hash = hashlib.md5(url.encode("utf-8")).hexdigest()
    return os.path.join(cache_dir, f"{url_hash}.json")


def load_cache(url: str) -> dict:
    cache_path = cache_file_path(url)
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_cache(url: str, data: dict):
    cache_path = cache_file_path(url)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def scrape_url(url: str, config: RedisConfigParams = {}) -> str:
    cache = Cache(config=config)
    cache_key = url
    cached_result = cache.get(cache_key)

    if cached_result:
        logger.log(f"scrape_url: Cache hit for", cache_key,
                   colors=["LOG", "BRIGHT_SUCCESS"])
        return cached_result

    logger.info(f"scrape_url: Cache miss for {cache_key}")
    url_scraper = UrlScraper()
    html_str = url_scraper.scrape_url(url)

    cache.set(cache_key, html_str)
    return html_str
