import ast
import asyncio
import json
import os
from datetime import datetime
from typing import TypedDict
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import httpx
import requests

from jet.cache.redis import RedisCache, RedisConfigParams
from jet.data.utils import generate_key
from jet.logger import logger
from jet.logger.timer import sleep_countdown
from jet.search.filters import deduplicate_results, filter_relevant, sort_by_score
from jet.search.formatters import decode_encoded_characters

DEFAULT_REDIS_PORT = 6379
DEFAULT_REDIS_DB = 1
DEFAULT_QUERY_URL = os.getenv("SEARXNG_URL")
DEFAULT_ENGINES = []


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
    publishedDate: str
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


def _normalize_csv_param(value, default: str = "", param_name: str = "param") -> str:
    """Normalize a multi-value parameter into a clean comma-separated string.
    Handles:
    - list/tuple → joined with commas
    - str that looks like a list repr ("['anime']") → parsed then joined
    - str CSV ("general, anime") → stripped and cleaned
    - None/empty → returns default
    SearXNG rejects Python list repr strings like "['anime']".
    This ensures only valid CSV strings reach build_query_url.
    """
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        result = ",".join(str(v).strip() for v in value if str(v).strip())
        return result or default
    if not isinstance(value, str):
        logger.warning(
            "🔧 %s: unexpected type %s, converting to string",
            param_name,
            type(value).__name__,
        )
        value = str(value)
    stripped = value.strip()
    if not stripped:
        return default
    if stripped.startswith("[") and stripped.endswith("]"):
        try:
            parsed = ast.literal_eval(stripped)
            if isinstance(parsed, (list, tuple)):
                result = ",".join(str(v).strip() for v in parsed if str(v).strip())
                logger.warning(
                    "🔧 Fixed malformed %s: %r → %r",
                    param_name,
                    stripped,
                    result,
                )
                return result or default
        except (ValueError, SyntaxError):
            logger.warning(
                "⚠️ %s looks like list repr but failed to parse: %r, treating as CSV",
                param_name,
                stripped,
            )
    result = ",".join(part.strip() for part in stripped.split(",") if part.strip())
    return result or default


def build_query_url(base_url: str, params: dict) -> str:
    """Helper function to construct the full search query URL with deduplicated parameters."""
    if not base_url:
        base_url = os.getenv("SEARXNG_URL", "http://localhost:8888")
        logger.warning(f"base_url was None/empty, using default: {base_url}")
    parsed_url = urlparse(base_url)
    path = parsed_url.path.rstrip("/")
    if not path or path == "/":
        path = "/search"
    scheme = parsed_url.scheme or "http"
    netloc = parsed_url.netloc or "localhost:8888"
    params_str = parsed_url.params or ""
    fragment = parsed_url.fragment or ""
    query_params = parse_qs(parsed_url.query)
    for key, value in params.items():
        if isinstance(value, (list, tuple)):
            query_params[key] = [str(v) for v in value]
        else:
            query_params[key] = [str(value)]
    encoded_params = {}
    for key, value_list in query_params.items():
        encoded_params[key] = value_list[0] if len(value_list) == 1 else value_list
    new_query = urlencode(encoded_params, doseq=True)
    new_url = urlunparse((scheme, netloc, path, params_str, new_query, fragment))
    logger.debug(f"Final URL: {new_url}")
    return new_url


def remove_empty_attributes(data):
    """Recursively remove keys with empty values from dictionaries and
    remove empty elements from lists."""
    if isinstance(data, dict):
        return {
            k: remove_empty_attributes(v)
            for k, v in data.items()
            if v not in [None, "", [], {}]
        }
    elif isinstance(data, list):
        return [remove_empty_attributes(v) for v in data if v not in [None, "", [], {}]]
    else:
        return data


def fetch_search_results(
    headers: dict, params: dict, query_url: str = DEFAULT_QUERY_URL
) -> QueryResponse:
    """Fetches search results from SearXNG."""
    logger.log("Requesting URL: ", query_url, colors=["LOG", "DEBUG"])
    response = requests.get(query_url, headers=headers)
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "")
    if "application/json" not in content_type:
        logger.error(
            f"Expected JSON but got {content_type}. "
            f"URL may be missing /search path: {query_url}"
        )
        raise ValueError(f"Non-JSON response from SearXNG: {content_type}")
    results = response.json()
    if not isinstance(results, dict) or "results" not in results:
        logger.error(f"Malformed SearXNG response: missing 'results' key")
        raise ValueError("Malformed SearXNG JSON response")
    for result in results.get("results", []):
        result["id"] = generate_key(result["url"])
    return results


def format_min_date(min_date: datetime) -> datetime:
    result = min_date.replace(hour=0, minute=0, second=0, microsecond=0)
    return result


def _fetch_with_retry(
    headers: dict,
    params: dict,
    query_url: str,
    max_retries: int,
) -> QueryResponse | None:
    """Fetch search results with retry logic. Returns None if all attempts yield no results."""
    RETRY_DELAY = 3
    result = None
    retries = 0
    while retries <= max_retries:
        if result and result.get("results", []):
            break
        try:
            result = fetch_search_results(headers, params, query_url)
            results_list = result.get("results", [])
            unresponsive = result.get("unresponsive_engines", [])
            if not results_list:
                if unresponsive and retries < max_retries:
                    logger.warning(
                        f"No results due to unresponsive engines {unresponsive}. "
                        f"Retrying {retries + 1}/{max_retries} after {RETRY_DELAY}s..."
                    )
                    sleep_countdown(RETRY_DELAY)
                    retries += 1
                    continue
                logger.info(
                    f"No results found for pageno={params.get('pageno', 1)}. "
                    f"This may be a valid empty result."
                )
                break
            else:
                break
        except requests.exceptions.RequestException as e:
            if retries < max_retries:
                logger.warning(
                    f"Request failed: {e}. Retrying {retries + 1}/{max_retries} after {RETRY_DELAY}s..."
                )
                sleep_countdown(RETRY_DELAY)
                retries += 1
                continue
            else:
                logger.error(f"Max retries reached. Error: {e}")
                return None
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
    max_retries: int = 1,
    **kwargs,
) -> list[SearchResult]:
    query = decode_encoded_characters(query)
    logger.debug("=== search_searxng called ===")
    logger.debug(f"query: {query}")
    logger.debug(f"query_url: {query_url}")
    logger.debug(f"count: {count}")
    logger.debug(f"min_score: {min_score}")
    logger.debug(f"min_date: {min_date}")
    logger.debug(
        f"engines (before processing): {engines} (type: {type(engines).__name__})"
    )
    logger.debug(f"include_sites: {include_sites}")
    logger.debug(f"exclude_sites: {exclude_sites}")
    logger.debug(f"max_retries: {max_retries}")
    logger.debug(f"kwargs: {json.dumps(kwargs, default=str)}")
    try:
        if include_sites:
            if isinstance(include_sites, str):
                include_sites = [
                    s.strip() for s in include_sites.split(",") if s.strip()
                ]
            include_query = " OR ".join([f"site:{site}" for site in include_sites])
            query += " " + include_query
        if exclude_sites:
            if isinstance(exclude_sites, str):
                exclude_sites = [
                    s.strip() for s in exclude_sites.split(",") if s.strip()
                ]
            exclude_query = " ".join([f"-site:{site}" for site in exclude_sites])
            query += " " + exclude_query
        raw_categories = kwargs.get("categories", ["general"])
        normalized_categories = _normalize_csv_param(
            raw_categories, default="general", param_name="categories"
        )
        params = {
            "q": query,
            "format": "json",
            "language": kwargs.get("language", "en"),
            "categories": normalized_categories,
        }
        current_pageno = kwargs.get("pageno", 1) or 1
        params["pageno"] = current_pageno
        if "safesearch" in kwargs:
            params["safesearch"] = kwargs["safesearch"] or 0
        if engines:
            normalized_engines = _normalize_csv_param(
                engines, default="", param_name="engines"
            )
            if normalized_engines:
                params["engines"] = normalized_engines
                logger.debug(
                    f"engines joined: {params['engines']} (type: {type(params['engines']).__name__})"
                )
        if not min_date:
            years_ago = kwargs.get("years_ago", 1)
            current_date = datetime.now()
            min_date = current_date.replace(year=current_date.year - years_ago)
        min_date = format_min_date(min_date)
        logger.debug("=== Params before build_query_url ===")
        for key, value in params.items():
            logger.debug(f"  {key}: {value} (type: {type(value).__name__})")
        query_url = build_query_url(query_url, params)
        logger.debug(f"=== Constructed query_url ===")
        logger.debug(f"query_url: {query_url}")
        headers = {"Accept": "application/json"}
        cached_result = None
        config = {"port": DEFAULT_REDIS_PORT, "db": DEFAULT_REDIS_DB, **config}
        cache = RedisCache(config=config)
        cache_key = query_url
        if use_cache:
            cached_result = cache.get(cache_key)
            if cached_result:
                cached_results = cached_result.get("results", [])
                if cached_results:
                    cached_count = len(cached_results)
                    if count is None or cached_count >= count:
                        logger.log(
                            "search_searxng: Cache hit for ",
                            cache_key,
                            colors=["SUCCESS", "BRIGHT_SUCCESS"],
                        )
                        logger.log(
                            f"search_searxng: Returning {cached_count} cached results",
                            colors=["SUCCESS", "BRIGHT_SUCCESS"],
                        )
                        return cached_results
                    else:
                        logger.warning(
                            f"search_searxng: Cache hit but insufficient results ({cached_count} < {count}) for {cache_key}"
                        )
                        cached_result = None
                else:
                    logger.warning(
                        f"search_searxng: Cache hit but contains empty results for {cache_key}. Clearing corrupted cache."
                    )
                    cache.clear(cache_key)
                    cached_result = None
            else:
                logger.warning(f"search_searxng: Cache miss for {cache_key}")
        result = cached_result
        if not result:
            result = _fetch_with_retry(headers, params, query_url, max_retries)
        if current_pageno == 1 and (not result or not result.get("results", [])):
            logger.info(
                "search_searxng: Page 1 returned no results. Falling back to page 2..."
            )
            page2_params = {**params, "pageno": 2}
            page2_url = build_query_url(query_url.rsplit("?", 1)[0], page2_params)
            page2_cache_key = page2_url
            page2_cached = None
            if use_cache:
                page2_cached = cache.get(page2_cache_key)
            if page2_cached and page2_cached.get("results", []):
                logger.log(
                    "search_searxng: Cache hit for page 2 fallback",
                    colors=["SUCCESS", "BRIGHT_SUCCESS"],
                )
                result = page2_cached
            else:
                page2_result = _fetch_with_retry(
                    headers, page2_params, page2_url, max_retries
                )
                if page2_result and page2_result.get("results", []):
                    result = page2_result
                    logger.info(
                        f"search_searxng: Page 2 fallback returned {len(result['results'])} results"
                    )
                else:
                    logger.warning(
                        "search_searxng: Page 2 fallback also returned no results"
                    )
        if not result or not result.get("results", []):
            logger.error(
                "search_searxng: No results after all retries and page 2 fallback, not caching"
            )
            return []
        result["number_of_results"] = len(result.get("results", []))
        result = remove_empty_attributes(result)
        results = result.get("results", [])
        results = filter_relevant(results, threshold=min_score)
        results = deduplicate_results(results)
        results = sort_by_score(results)
        results = results[:count] if count is not None else results
        result["results"] = results
        if results:
            effective_cache_key = cache_key if current_pageno == 1 else query_url
            cache.set(effective_cache_key, result)
            logger.log(
                f"search_searxng: Cached {len(results)} results for {effective_cache_key}",
                colors=["SUCCESS", "BRIGHT_SUCCESS"],
            )
        else:
            logger.warning(
                f"search_searxng: No valid results after filtering for {cache_key}. Not caching."
            )
            cache.clear(cache_key)
        return results
    except (KeyError, TypeError) as e:
        logger.error(f"Error in search_searxng: {e}")
        return []


# =========================================================================
# Async equivalent of search_searxng
# =========================================================================


async def _async_fetch_with_retry(
    query_url: str,
    params: dict,
    max_retries: int = 1,
    timeout: float = 12.0,
) -> QueryResponse | None:
    """Async fetch with retry logic. Mirrors _fetch_with_retry but non-blocking."""
    RETRY_DELAY = 3
    result = None
    retries = 0

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        while retries <= max_retries:
            if result and result.get("results", []):
                break
            try:
                headers = {"Accept": "application/json"}
                response = await client.get(query_url, headers=headers)
                response.raise_for_status()

                content_type = response.headers.get("Content-Type", "")
                if "application/json" not in content_type:
                    logger.error(f"async_fetch: expected JSON but got {content_type}")
                    raise ValueError(f"Non-JSON response: {content_type}")

                result = response.json()
                if not isinstance(result, dict) or "results" not in result:
                    logger.error("async_fetch: malformed JSON response")
                    raise ValueError("Malformed SearXNG JSON response")

                for r in result.get("results", []):
                    r["id"] = generate_key(r["url"])

                results_list = result.get("results", [])
                unresponsive = result.get("unresponsive_engines", [])

                if not results_list:
                    if unresponsive and retries < max_retries:
                        logger.warning(
                            f"async_fetch: unresponsive engines {unresponsive}, "
                            f"retrying {retries + 1}/{max_retries}"
                        )
                        await asyncio.sleep(RETRY_DELAY)
                        retries += 1
                        continue
                    logger.info(
                        f"async_fetch: no results for pageno={params.get('pageno', 1)}"
                    )
                    break
                else:
                    break

            except (httpx.RequestError, httpx.HTTPStatusError, ValueError) as e:
                if retries < max_retries:
                    logger.warning(
                        f"async_fetch: {type(e).__name__}: {e}, "
                        f"retrying {retries + 1}/{max_retries}"
                    )
                    await asyncio.sleep(RETRY_DELAY)
                    retries += 1
                    continue
                else:
                    logger.error(f"async_fetch: max retries reached - {e}")
                    return None

    return result


async def async_search_searxng(
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
    max_retries: int = 1,
    timeout: float = 12.0,
    **kwargs,
) -> list[SearchResult]:
    """Async equivalent of search_searxng using httpx.AsyncClient.

    Provides identical functionality to search_searxng (Redis caching,
    result filtering, deduplication, score sorting, page-2 fallback)
    but uses non-blocking I/O for use in async pipelines like crawl4ai_lib.

    Args:
        query: Search query string.
        query_url: SearXNG base URL. Defaults to SEARXNG_URL env var.
        count: Maximum number of results to return.
        min_score: Minimum relevance score threshold for filtering.
        min_date: Minimum publication date filter.
        config: Redis configuration overrides.
        use_cache: Whether to use Redis cache.
        engines: Specific search engines to use.
        include_sites: Limit search to these domains.
        exclude_sites: Exclude these domains from search.
        max_retries: Maximum retry attempts on failure or empty results.
        timeout: HTTP request timeout in seconds.
        **kwargs: Additional params forwarded to SearXNG (language,
            categories, pageno, safesearch, years_ago).

    Returns:
        List of SearchResult dicts sorted by score (descending).
    """
    query = decode_encoded_characters(query)
    logger.info(f"async_search_searxng: query='{query[:80]}...', count={count}")

    try:
        # --- Build query with site filters (mirrors sync version) ---
        if include_sites:
            if isinstance(include_sites, str):
                include_sites = [
                    s.strip() for s in include_sites.split(",") if s.strip()
                ]
            include_query = " OR ".join([f"site:{site}" for site in include_sites])
            query += " " + include_query

        if exclude_sites:
            if isinstance(exclude_sites, str):
                exclude_sites = [
                    s.strip() for s in exclude_sites.split(",") if s.strip()
                ]
            exclude_query = " ".join([f"-site:{site}" for site in exclude_sites])
            query += " " + exclude_query

        # --- Build params (mirrors sync version) ---
        raw_categories = kwargs.get("categories", ["general"])
        normalized_categories = _normalize_csv_param(
            raw_categories, default="general", param_name="categories"
        )
        params = {
            "q": query,
            "format": "json",
            "language": kwargs.get("language", "en"),
            "categories": normalized_categories,
        }
        current_pageno = kwargs.get("pageno", 1) or 1
        params["pageno"] = current_pageno

        if "safesearch" in kwargs:
            params["safesearch"] = kwargs["safesearch"] or 0

        if engines:
            normalized_engines = _normalize_csv_param(
                engines, default="", param_name="engines"
            )
            if normalized_engines:
                params["engines"] = normalized_engines

        if not min_date:
            years_ago = kwargs.get("years_ago", 1)
            current_date = datetime.now()
            min_date = current_date.replace(year=current_date.year - years_ago)
        min_date = format_min_date(min_date)

        full_url = build_query_url(query_url, params)
        logger.debug(f"async_search_searxng: full_url={full_url}")

        # --- Redis cache check (identical logic to sync version) ---
        redis_config = {"port": DEFAULT_REDIS_PORT, "db": DEFAULT_REDIS_DB, **config}
        cache = RedisCache(config=redis_config)
        cache_key = full_url

        if use_cache:
            cached_result = cache.get(cache_key)
            if cached_result:
                cached_results = cached_result.get("results", [])
                if cached_results:
                    cached_count = len(cached_results)
                    if count is None or cached_count >= count:
                        logger.info(
                            f"async_search_searxng: cache hit, "
                            f"returning {cached_count} results"
                        )
                        return cached_results
                    else:
                        logger.warning(
                            f"async_search_searxng: cache hit but insufficient "
                            f"results ({cached_count} < {count})"
                        )
                else:
                    logger.warning(
                        f"async_search_searxng: cache hit but empty, clearing"
                    )
                    cache.clear(cache_key)

        # --- Async fetch with retry ---
        result = await _async_fetch_with_retry(
            full_url, params, max_retries=max_retries, timeout=timeout
        )

        # --- Page 2 fallback (mirrors sync version) ---
        if current_pageno == 1 and (not result or not result.get("results", [])):
            logger.info("async_search_searxng: page 1 empty, falling back to page 2")
            page2_params = {**params, "pageno": 2}
            page2_url = build_query_url(full_url.rsplit("?", 1)[0], page2_params)
            page2_cache_key = page2_url

            page2_cached = None
            if use_cache:
                page2_cached = cache.get(page2_cache_key)

            if page2_cached and page2_cached.get("results", []):
                logger.info("async_search_searxng: page 2 cache hit")
                result = page2_cached
            else:
                page2_result = await _async_fetch_with_retry(
                    page2_url,
                    page2_params,
                    max_retries=max_retries,
                    timeout=timeout,
                )
                if page2_result and page2_result.get("results", []):
                    result = page2_result
                    logger.info(
                        f"async_search_searxng: page 2 returned "
                        f"{len(result['results'])} results"
                    )

        if not result or not result.get("results", []):
            logger.error("async_search_searxng: no results after retries and fallback")
            return []

        # --- Post-processing (identical to sync version) ---
        result["number_of_results"] = len(result.get("results", []))
        result = remove_empty_attributes(result)

        results = result.get("results", [])
        results = filter_relevant(results, threshold=min_score)
        results = deduplicate_results(results)
        results = sort_by_score(results)
        results = results[:count] if count is not None else results
        result["results"] = results

        # --- Cache store ---
        if results:
            effective_cache_key = cache_key if current_pageno == 1 else full_url
            cache.set(effective_cache_key, result)
            logger.info(f"async_search_searxng: cached {len(results)} results")
        else:
            logger.warning("async_search_searxng: no valid results after filtering")
            cache.clear(cache_key)

        logger.info(f"async_search_searxng: returning {len(results)} results")
        return results

    except (KeyError, TypeError) as e:
        logger.error(f"async_search_searxng: error - {e}")
        return []


if __name__ == "__main__":
    import argparse

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
        "--max-retries",
        type=int,
        default=1,
        help="Maximum number of retry attempts",
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
