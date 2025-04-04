from urllib.parse import urlparse
from datetime import datetime
import json
from .cache import RedisCache
import requests
from typing import Optional, TypedDict
from urllib.parse import urlencode
from pydantic import BaseModel
from jet.logger import logger
from .filters import filter_relevant, filter_by_date, deduplicate_results, sort_by_score
from .formatters import decode_encoded_characters
from jet.cache.redis import RedisConfigParams


DEFAULT_REDIS_PORT = 3101


class SearchResult(TypedDict):
    url: str
    title: str
    content: str
    engine: str
    template: str
    parsed_url: list[str]
    engines: list[str]
    positions: list[int]
    publishedDate: str  # Alternatively, use datetime if you plan to parse it
    score: float
    category: str


class QueryResponse(TypedDict, total=False):
    query: str
    number_of_results: int
    results: list[SearchResult]
    answers: list[str]
    corrections: list[str]
    infoboxes: list[str]
    suggestions: list[str]
    unresponsive_engines: list[str]


class NoResultsFoundError(Exception):
    """Custom exception to be raised when no results are found."""
    pass


def build_query_url(base_url: str, params: dict) -> str:
    """Helper function to construct the full search query URL."""
    encoded_params = urlencode(params)
    return f"{base_url.split('?')[0]}?{encoded_params}"


def remove_empty_attributes(data):
    """
    Recursively remove keys with empty values from dictionaries and 
    remove empty elements from lists.
    """
    if isinstance(data, dict):
        # Return a new dictionary with only non-empty values
        return {k: remove_empty_attributes(v) for k, v in data.items() if v not in [None, "", [], {}]}
    elif isinstance(data, list):
        # Return a new list with non-empty elements
        return [remove_empty_attributes(v) for v in data if v not in [None, "", [], {}]]
    else:
        # Return the data as is if it's not a dict or list
        return data


def fetch_search_results(query_url: str, headers: dict, params: dict) -> QueryResponse:
    """Fetches search results from SearXNG."""

    logger.log("Requesting URL:", query_url, colors=["LOG", "DEBUG"])
    logger.log("Headers:")
    logger.info(json.dumps(headers, indent=2))
    logger.log("Params:")
    logger.info(json.dumps(params, indent=2))
    response = requests.get(query_url, headers=headers, params=params)
    response.raise_for_status()
    return response.json()


def format_min_date(min_date: datetime) -> datetime:
    # hours, minutes, and seconds set to 0
    result = min_date.replace(hour=0,
                              minute=0, second=0, microsecond=0)
    return result


def filter_unique_hosts(results: list[SearchResult]) -> list[SearchResult]:
    """
    Filter results to ensure unique hosts with the highest score.
    """
    host_map = {}
    for result in results:
        host = urlparse(result["url"]).netloc
        if host not in host_map or result["score"] > host_map[host]["score"]:
            host_map[host] = result
    return list(host_map.values())


def search_searxng(query_url: str, query: str, count: Optional[int] = None, min_score: float = 0.2, min_date: Optional[datetime] = None, config: RedisConfigParams = {}, use_cache: bool = True, include_sites: Optional[list[str]] = None, exclude_sites: Optional[list[str]] = None, **kwargs) -> list[SearchResult]:
    query = decode_encoded_characters(query)
    try:
        # Add the include_sites filter if provided
        if include_sites:
            include_query = " OR ".join(
                [f"site:{site}" for site in include_sites])
            query += " " + include_query

        # Add the exclude_sites filter if provided
        if exclude_sites:
            exclude_query = " ".join(
                [f"-site:{site}" for site in exclude_sites])
            query += " " + exclude_query

        # Start building the base query params
        params = {
            "q": query,
            "format": "json",
            "pageno": kwargs.get("pageno", 1),
            "safesearch": kwargs.get("safesearch", 2),
            "language": kwargs.get("language", "en"),
            "categories": ",".join(kwargs.get("categories", ["general"])),
            "engines": ",".join(kwargs.get("engines", ["google", "brave", "duckduckgo", "bing", "yahoo"])),
        }

        # Handling min_date (optional)
        if not min_date:
            years_ago = kwargs.get("years_ago", 1)
            current_date = datetime.now()
            min_date = current_date.replace(year=current_date.year - years_ago)
        min_date = format_min_date(min_date)
        min_date_iso = min_date.isoformat()

        # Prepare the query URL
        query_url = build_query_url(query_url, params)
        headers = {"Accept": "application/json"}

        cached_result = None

        if use_cache:
            config = {"port": DEFAULT_REDIS_PORT, **config}
            cache = RedisCache(config=config)
            cache_key = query_url
            cached_result = cache.get(cache_key)

            if cached_result:
                logger.log(f"search_searxng: Cache hit for",
                           cache_key, colors=["SUCCESS", "BRIGHT_SUCCESS"])
                # return cached_result["results"]
            else:
                logger.warning(f"search_searxng: Cache miss for {cache_key}")

        # Fetch search results
        result = cached_result or fetch_search_results(
            query_url, headers, params)
        result['number_of_results'] = len(result.get("results", []))
        result = remove_empty_attributes(result)

        # Filter and sort results
        results = result.get("results", [])
        results = filter_relevant(results, threshold=min_score)
        results = deduplicate_results(results)
        results = sort_by_score(results)
        results = filter_unique_hosts(results)
        results = results[:count] if count is not None else results
        result["results"] = results

        # Cache the result if caching is enabled
        if use_cache:
            cache.set(cache_key, result)
        return results

    except (requests.exceptions.RequestException, KeyError, TypeError) as e:
        logger.error(f"Error in search_searxng: {e}")
        return []
