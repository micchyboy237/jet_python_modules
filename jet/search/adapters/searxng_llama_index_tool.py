import time
from typing import List, Dict, Optional, TypedDict
from datetime import datetime
import requests
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from jet.logger import logger
from jet.search.searxng import build_query_url, fetch_search_results, SearchResult, QueryResponse, format_min_date, remove_empty_attributes
from jet.search.formatters import decode_encoded_characters
from jet.search.filters import filter_relevant, filter_by_date, deduplicate_results, sort_by_score
from jet.cache.redis import RedisConfigParams, RedisCache
from jet.data.utils import generate_key

DEFAULT_REDIS_PORT = 3101
DEFAULT_URL = "http://jethros-macbook-air.local:3000/search"


class SearXNGSearchToolSpec(BaseToolSpec):
    """SearXNGSearch tool spec."""

    spec_functions = ["searxng_instant_search", "searxng_full_search"]

    def __init__(self, base_url: str = DEFAULT_URL, redis_config: Optional[RedisConfigParams] = None) -> None:
        """
        Initialize the SearXNG search tool with a base URL and optional Redis configuration.

        Args:
            base_url (str): The base URL for the SearXNG instance (e.g., 'https://searxng.example.com').
            redis_config (Optional[RedisConfigParams]): Configuration for Redis caching.
        """
        self.base_url = base_url.rstrip('/')
        self.redis_config = redis_config or {"port": DEFAULT_REDIS_PORT}
        self.cache = RedisCache(config=self.redis_config)
        super().__init__()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(Exception),
        before_sleep=lambda retry_state: logger.info(
            f"Rate limit hit, retrying after {retry_state.next_action.sleep} seconds..."
        )
    )
    def searxng_instant_search(self, query: str) -> List[Dict]:
        """
        Make a query to SearXNG to receive instant answers.

        Args:
            query (str): The query to be passed to SearXNG.

        Returns:
            List[Dict]: List of instant answer results.

        Raises:
            Exception: If the API call fails after retries.
        """
        try:
            query = decode_encoded_characters(query)
            params = {
                "q": query,
                "format": "json",
                "categories": "general",
                "engines": "google,duckduckgo",
            }
            query_url = build_query_url(self.base_url, params)
            headers = {"Accept": "application/json"}
            logger.info(f"Executing instant search for query: {query}")
            result = fetch_search_results(query_url, headers, params)
            answers = result.get("answers", [])
            return [{"text": answer} for answer in answers]
        except Exception as e:
            logger.error(
                f"Instant search failed for query '{query}': {str(e)}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(Exception),
        before_sleep=lambda retry_state: logger.info(
            f"Rate limit hit, retrying after {retry_state.next_action.sleep} seconds..."
        )
    )
    def searxng_full_search(
        self,
        query: str,
        count: Optional[int] = 10,
    ) -> List[SearchResult]:
        """
        Make a query to SearXNG to receive full search results.
        Args:
            query (str): The query to be passed to SearXNG.
            count (Optional[int]): Maximum number of results to return.
        Returns:
            List[SearchResult]: List of filtered and sorted search results.
        Raises:
            Exception: If the API call fails after retries.
        """
        try:
            query = decode_encoded_characters(query)
            params = {
                "q": query,
                "format": "json",
                "pageno": 1,
                "safesearch": 2,
                "language": "en",
                "categories": "general",
                "engines": "google,duckduckgo",
            }
            query_url = build_query_url(self.base_url, params)
            headers = {"Accept": "application/json"}
            cache_key = query_url
            cached_result = self.cache.get(cache_key)
            if cached_result and cached_result.get("results", []):
                cached_count = len(cached_result["results"])
                if count is None or cached_count >= count:
                    logger.info(f"Cache hit for {cache_key}")
                    return cached_result["results"][:count]
                else:
                    logger.warning(
                        f"Cache hit but insufficient results ({cached_count} < {count}) for {cache_key}")
            result = fetch_search_results(query_url, headers, params)
            if not result.get("results", []):
                logger.warning("No results found for query")
                return []
            results = result.get("results", [])
            results = filter_relevant(results, threshold=0.1)
            results = deduplicate_results(results)
            results = sort_by_score(results)
            results = results[:count] if count is not None else results
            cache_data = {"results": results,
                          "number_of_results": len(results)}
            self.cache.set(cache_key, cache_data)
            logger.info(
                f"Full search returned {len(results)} results for query: {query}")
            return results
        except Exception as e:
            logger.error(f"Full search failed for query '{query}': {str(e)}")
            raise
