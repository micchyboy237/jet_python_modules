import json
import os
import time
from datetime import datetime
from typing import TypedDict
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import requests

from jet.cache.redis import RedisCache, RedisConfigParams
from jet.data.utils import generate_key
from jet.logger import logger
from jet.search.filters import deduplicate_results, filter_relevant, sort_by_score
from jet.search.formatters import decode_encoded_characters

# DEFAULT_REDIS_PORT = 3101
DEFAULT_REDIS_PORT = 6379
DEFAULT_REDIS_DB = 1
DEFAULT_QUERY_URL = os.getenv("SEARXNG_URL")
DEFAULT_ENGINES = [
    # "google",
    # "brave",
    # "yahoo",
    # "mojeek",        # Fully independent crawler
    # "brave",         # Brave Search – own index + high reliability
    # "startpage",     # Anonymous Google proxy, very stable
    # "kagi",          # Premium paid engine, excellent quality & uptime
    # "duckduckgo",    # Widely trusted metasearch with great coverage
    # "qwant",         # Strong European privacy-focused engine
    # "swisscows",     # Privacy-first, family-safe, reliable Bing backend
]


class SearchResult(TypedDict):
    id: str
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
    """Helper function to construct the full search query URL with deduplicated parameters."""
    # Parse the base URL to separate the query string
    parsed_url = urlparse(base_url)
    query_params = parse_qs(parsed_url.query)

    # Merge the existing query parameters with the new params
    # If a key exists in both, the value from `params` will overwrite the existing one
    for key, value in params.items():
        if isinstance(value, (list, tuple)):
            query_params[key] = list(value)
        else:
            query_params[key] = [value]

    # Convert the query_params back to a string
    # urlencode expects a dictionary where values are either strings or lists of strings
    # We need to ensure that single-value keys are not wrapped in a list
    encoded_params = {}
    for key, value_list in query_params.items():
        if len(value_list) == 1:
            encoded_params[key] = value_list[0]
        else:
            encoded_params[key] = value_list

    # Rebuild the URL with the deduplicated and merged parameters
    new_query = urlencode(encoded_params, doseq=True)
    new_url = urlunparse(
        (
            parsed_url.scheme,
            parsed_url.netloc,
            parsed_url.path,
            parsed_url.params,
            new_query,
            parsed_url.fragment,
        )
    )

    return new_url


def remove_empty_attributes(data):
    """
    Recursively remove keys with empty values from dictionaries and
    remove empty elements from lists.
    """
    if isinstance(data, dict):
        # Return a new dictionary with only non-empty values
        return {
            k: remove_empty_attributes(v)
            for k, v in data.items()
            if v not in [None, "", [], {}]
        }
    elif isinstance(data, list):
        # Return a new list with non-empty elements
        return [remove_empty_attributes(v) for v in data if v not in [None, "", [], {}]]
    else:
        # Return the data as is if it's not a dict or list
        return data


def fetch_search_results(
    headers: dict, params: dict, query_url: str = DEFAULT_QUERY_URL
) -> QueryResponse:
    """Fetches search results from SearXNG."""

    logger.log("Requesting URL: ", query_url, colors=["LOG", "DEBUG"])
    logger.log("Headers:")
    logger.info(json.dumps(headers, indent=2))
    logger.log("Params:")
    logger.info(json.dumps(params, indent=2))
    response = requests.get(query_url, headers=headers, params=params)
    response.raise_for_status()
    results = response.json()

    # Add id to each result using generate_key based on URL
    for result in results.get("results", []):
        result["id"] = generate_key(result["url"])

    return results


def format_min_date(min_date: datetime) -> datetime:
    # hours, minutes, and seconds set to 0
    result = min_date.replace(hour=0, minute=0, second=0, microsecond=0)
    return result


def search_searxng(
    query: str,
    query_url: str = DEFAULT_QUERY_URL,
    count: int | None = None,
    min_score: float = 0.1,
    min_date: datetime | None = None,
    config: RedisConfigParams = {},
    use_cache: bool = True,
    engines: list[str] | None = DEFAULT_ENGINES,
    include_sites: list[str] | None = None,
    exclude_sites: list[str] | None = None,
    max_retries: int = 3,
    **kwargs,
) -> list[SearchResult]:
    query = decode_encoded_characters(query)
    try:
        # Add the include_sites filter if provided
        if include_sites:
            include_query = " OR ".join([f"site:{site}" for site in include_sites])
            query += " " + include_query

        # Add the exclude_sites filter if provided
        if exclude_sites:
            exclude_query = " ".join([f"-site:{site}" for site in exclude_sites])
            query += " " + exclude_query

        # Start building the base query params
        params = {
            "q": query,
            "format": "json",
            "language": kwargs.get("language", "en"),
            "categories": ",".join(kwargs.get("categories", ["general"])),
        }

        if "pageno" in kwargs:
            params["pageno"] = kwargs["pageno"] or 1

        if "safesearch" in kwargs:
            params["safesearch"] = kwargs["safesearch"] or 0

        if engines:
            params["engines"] = (",".join(engines),)

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

        config = {"port": DEFAULT_REDIS_PORT, "db": DEFAULT_REDIS_DB, **config}
        cache = RedisCache(config=config)
        cache_key = query_url

        if use_cache:
            cached_result = cache.get(cache_key)

            if cached_result and cached_result.get("results", []):
                cached_count = len(cached_result["results"])
                if count is None or cached_count >= count:
                    logger.log(
                        "search_searxng: Cache hit for ",
                        cache_key,
                        colors=["SUCCESS", "BRIGHT_SUCCESS"],
                    )
                else:
                    logger.warning(
                        f"search_searxng: Cache hit but insufficient results ({cached_count} < {count}) for {cache_key}"
                    )
                    cached_result = None
            else:
                logger.warning(
                    f"search_searxng: Cache miss or empty results for {cache_key}"
                )

        # Fetch search results with retries
        result = cached_result
        retries = 0
        while retries <= max_retries:
            if result and result.get("results", []):
                break

            try:
                result = fetch_search_results(headers, params, query_url)
                if not result.get("results", []):
                    if retries < max_retries:
                        delay = 2**retries  # Exponential backoff: 1s, 2s, 4s
                        logger.warning(
                            f"No results found. Retrying {retries + 1}/{max_retries} after {delay}s delay..."
                        )
                        time.sleep(delay)
                        retries += 1
                        continue
                    else:
                        logger.error("Max retries reached with no results.")
                        return []
            except requests.exceptions.RequestException as e:
                if retries < max_retries:
                    delay = 2**retries  # Exponential backoff: 1s, 2s, 4s
                    logger.warning(
                        f"Request failed: {e}. Retrying {retries + 1}/{max_retries} after {delay}s delay..."
                    )
                    time.sleep(delay)
                    retries += 1
                    continue
                else:
                    logger.error(f"Max retries reached. Error: {e}")
                    return []

        result["number_of_results"] = len(result.get("results", []))
        result = remove_empty_attributes(result)

        # Filter and sort results
        results = result.get("results", [])
        results = filter_relevant(results, threshold=min_score)
        results = deduplicate_results(results)
        results = sort_by_score(results)
        results = results[:count] if count is not None else results
        result["results"] = results

        # Cache the result
        cache.set(cache_key, result)
        return results

    except (KeyError, TypeError) as e:
        logger.error(f"Error in search_searxng: {e}")
        return []


if __name__ == "__main__":
    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser(description="Search using SearXNG")
    parser.add_argument("query", help="Search query string")
    parser.add_argument(
        "--count", type=int, default=None, help="Maximum number of results to return"
    )
    parser.add_argument(
        "--min-score", type=float, default=0.1, help="Minimum relevance score threshold"
    )
    parser.add_argument(
        "--min-date",
        type=str,
        default=None,
        help="Minimum date filter (ISO format: YYYY-MM-DD)",
    )
    parser.add_argument(
        "--use-cache", action="store_true", default=True, help="Use Redis cache"
    )
    parser.add_argument(
        "--no-cache", action="store_false", dest="use_cache", help="Disable Redis cache"
    )
    parser.add_argument(
        "--engines", nargs="*", default=None, help="Search engines to use"
    )
    parser.add_argument(
        "--include-sites",
        nargs="*",
        default=None,
        help="Limit search to specific sites",
    )
    parser.add_argument(
        "--exclude-sites",
        nargs="*",
        default=None,
        help="Exclude specific sites from search",
    )
    parser.add_argument(
        "--max-retries", type=int, default=3, help="Maximum number of retry attempts"
    )
    parser.add_argument("--language", default="en", help="Search language")
    parser.add_argument(
        "--categories", nargs="*", default=["general"], help="Search categories"
    )
    parser.add_argument("--pageno", type=int, default=1, help="Page number")
    parser.add_argument(
        "--safesearch", type=int, default=0, help="Safe search level (0, 1, or 2)"
    )
    parser.add_argument(
        "--years-ago",
        type=int,
        default=1,
        help="Number of years ago for default min_date",
    )
    parser.add_argument(
        "--output", choices=["json", "text"], default="text", help="Output format"
    )

    args = parser.parse_args()

    # Parse min_date if provided
    min_date = None
    if args.min_date:
        min_date = datetime.fromisoformat(args.min_date)

    results = search_searxng(
        query=args.query,
        count=args.count,
        min_score=args.min_score,
        min_date=min_date,
        use_cache=args.use_cache,
        engines=args.engines,
        include_sites=args.include_sites,
        exclude_sites=args.exclude_sites,
        max_retries=args.max_retries,
        language=args.language,
        categories=args.categories,
        pageno=args.pageno,
        safesearch=args.safesearch,
        years_ago=args.years_ago,
    )

    if args.output == "json":
        print(json.dumps(results, indent=2))
    else:
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['title']}")
            print(f"   URL: {result['url']}")
            print(f"   Score: {result.get('score', 'N/A')}")
            print(f"   Content: {result.get('content', '')[:200]}...")
            print()
