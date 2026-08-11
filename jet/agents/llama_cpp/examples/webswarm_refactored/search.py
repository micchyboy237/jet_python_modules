import asyncio
import logging

from jet.search.searxng import search_searxng

from .config import (
    SEARXNG_CATEGORIES,
    SEARXNG_ENGINES,
    SEARXNG_MAX_RESULTS,
    SEARXNG_MIN_SCORE,
    SEARXNG_QUERY_URL,
    SEARXNG_USE_CACHE,
)

logger = logging.getLogger("webswarm")


async def web_search(query: str) -> list[str]:
    """Async wrapper around jet.search.searxng.search_searxng."""
    loop = asyncio.get_running_loop()
    try:
        results = await loop.run_in_executor(
            None,
            lambda: search_searxng(
                query=query,
                query_url=SEARXNG_QUERY_URL,
                count=SEARXNG_MAX_RESULTS,
                min_score=SEARXNG_MIN_SCORE,
                engines=SEARXNG_ENGINES,
                categories=SEARXNG_CATEGORIES,
                use_cache=SEARXNG_USE_CACHE,
            ),
        )
        urls = [r["url"] for r in results if r.get("url")]
        logger.info(f"SearXNG returned {len(urls)} URLs for: {query[:80]}")
        return urls
    except Exception as e:
        logger.error(f"SearXNG search failed for '{query[:80]}': {e}")
        return []
