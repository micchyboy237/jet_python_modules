import ast
import json
import os
from datetime import datetime
from typing import TypedDict
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

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

    # Case 1: Already a proper list/tuple
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

    # Case 2: String that looks like a Python list repr: "['anime']" or '["general","news"]'
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

    # Case 3: Normal CSV string — clean up whitespace around commas
    result = ",".join(part.strip() for part in stripped.split(",") if part.strip())
    return result or default


def build_query_url(base_url: str, params: dict) -> str:
    """Helper function to construct the full search query URL with deduplicated parameters."""
    if not base_url:
        base_url = os.getenv("SEARXNG_URL", "http://localhost:8888")
        logger.warning(f"base_url was None/empty, using default: {base_url}")

    logger.debug("=== build_query_url called ===")
    logger.debug(f"base_url: {base_url}")
    logger.debug(f"params: {json.dumps(params, default=str)}")

    parsed_url = urlparse(base_url)
    scheme = (
        parsed_url.scheme.decode()
        if isinstance(parsed_url.scheme, bytes)
        else parsed_url.scheme or "http"
    )
    netloc = (
        parsed_url.netloc.decode()
        if isinstance(parsed_url.netloc, bytes)
        else parsed_url.netloc or "localhost:8888"
    )
    path = (
        parsed_url.path.decode()
        if isinstance(parsed_url.path, bytes)
        else parsed_url.path or "/"
    )
    params_str = (
        parsed_url.params.decode()
        if isinstance(parsed_url.params, bytes)
        else parsed_url.params or ""
    )
    fragment = (
        parsed_url.fragment.decode()
        if isinstance(parsed_url.fragment, bytes)
        else parsed_url.fragment or ""
    )

    logger.debug(
        f"parsed_url: scheme={scheme}, netloc={netloc}, path={path}, query={parsed_url.query}"
    )

    query_params = parse_qs(parsed_url.query)
    logger.debug(f"initial query_params: {json.dumps(query_params, default=str)}")

    for key, value in params.items():
        logger.debug(f"Processing param: {key}={value} (type: {type(value).__name__})")
        if isinstance(value, (list, tuple)):
            converted = [str(v) for v in value]
            logger.debug(f"  Converted list/tuple to: {converted}")
            query_params[key] = converted
        else:
            logger.debug(f"  Converted single value to: [str(value)]")
            query_params[key] = [str(value)]

    encoded_params = {}
    for key, value_list in query_params.items():
        logger.debug(f"Encoding: {key} = {value_list}")
        if len(value_list) == 1:
            encoded_params[key] = value_list[0]
        else:
            encoded_params[key] = value_list

    logger.debug(f"encoded_params: {json.dumps(encoded_params, default=str)}")

    new_query = urlencode(encoded_params, doseq=True)
    logger.debug(f"new_query string: {new_query}")

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
    logger.log("Headers:")
    logger.info(json.dumps(headers, indent=2))
    logger.log("Params (already embedded in URL):")
    logger.info(json.dumps(params, indent=2))

    response = requests.get(query_url, headers=headers)
    response.raise_for_status()
    results = response.json()

    for result in results.get("results", []):
        result["id"] = generate_key(result["url"])

    return results


def format_min_date(min_date: datetime) -> datetime:
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

    # ===== DEBUG: Log all input parameters =====
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
        # ✅ Normalize include/exclude sites regardless of input type
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

        # ✅ FIXED: Normalize ALL multi-value params through _normalize_csv_param
        # Prevents SearXNG ValidationException from list repr strings like "['anime']"
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

        if "pageno" in kwargs:
            params["pageno"] = kwargs["pageno"] or 1
        if "safesearch" in kwargs:
            params["safesearch"] = kwargs["safesearch"] or 0

        # ✅ FIXED: Normalize engines through same helper
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
        min_date_iso = min_date.isoformat()

        # ===== DEBUG: Log params before build_query_url =====
        logger.debug("=== Params before build_query_url ===")
        for key, value in params.items():
            logger.debug(f"  {key}: {value} (type: {type(value).__name__})")

        query_url = build_query_url(query_url, params)

        # ===== DEBUG: Log constructed URL =====
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
        retries = 0
        while retries <= max_retries:
            if result and result.get("results", []):
                break
            try:
                result = fetch_search_results(headers, params, query_url)
                if not result.get("results", []):
                    if retries < max_retries:
                        delay = 10 * (2**retries)
                        logger.warning(
                            f"No results found. Retrying {retries + 1}/{max_retries} after {delay}s delay..."
                        )
                        sleep_countdown(delay)
                        retries += 1
                        continue
                    else:
                        logger.error("Max retries reached with no results.")
                        return []
                else:
                    break
            except requests.exceptions.RequestException as e:
                if retries < max_retries:
                    delay = 10 * (2**retries)
                    logger.warning(
                        f"Request failed: {e}. Retrying {retries + 1}/{max_retries} after {delay}s delay..."
                    )
                    sleep_countdown(delay)
                    retries += 1
                    continue
                else:
                    logger.error(f"Max retries reached. Error: {e}")
                    return []

        if not result or not result.get("results", []):
            logger.error("search_searxng: No results after all retries, not caching")
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
            cache.set(cache_key, result)
            logger.log(
                f"search_searxng: Cached {len(results)} results for {cache_key}",
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
