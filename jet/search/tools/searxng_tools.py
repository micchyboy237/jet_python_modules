import os
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict

from jet.logger import logger
from jet.search.searxng import search_searxng

SEARCH_URL = os.getenv("SEARXNG_URL")


class WebSearchResult(TypedDict):
    url: str
    title: str
    content: str
    score: float


def search_web(
    query: str, count: Optional[int] = 10, config: Optional[Dict[str, Any]] = None
) -> List[WebSearchResult]:
    """
    Performs a web search using the SearXNG engine and returns filtered results.

    This function is designed as a tool for LLMs to perform web searches with a simplified interface.
    It leverages the search_searxng function to fetch and filter results.

    Args:
        query (str): The search query string.
        count (Optional[int], optional): Maximum number of results to return. Defaults to 10.
        config (Optional[Dict[str, Any]], optional): Configuration dictionary for advanced parameters.
            Supported keys:
            - min_date (datetime): Minimum publication date for results (default: None).
            - include_sites (List[str]): Sites to include in the search (default: None).
            - exclude_sites (List[str]): Sites to exclude from the search (default: None).

    Returns:
        List[WebSearchResult]: A list of filtered search results, each containing url, title, content, and score.

    Example:
        results = search_web(
            query="python programming",
            count=5,
            config={"include_sites": ["python.org"], "min_date": datetime(2023, 1, 1)}
        )
    """
    logger.debug(f"Starting web search with query: {query}, count: {count}")
    config = config or {}
    logger.debug(f"Configuration used: {config}")

    results = search_searxng(
        query_url=SEARCH_URL,
        query=query,
        count=int(count),
        min_date=config.get("min_date"),
        config={},
        include_sites=config.get("include_sites"),
        exclude_sites=config.get("exclude_sites"),
    )

    logger.debug(f"Search completed, retrieved {len(results)} results")

    # Filter results to include only those with all required keys
    required_keys = {"url", "title", "content", "score"}
    valid_results = [
        result for result in results if all(key in result for key in required_keys)
    ]

    logger.debug(
        f"Filtered to {len(valid_results)} results with all required keys (url, title, content, score)"
    )

    # Format valid results to include only url, title, content, and score
    formatted_results: List[WebSearchResult] = [
        {
            "url": result["url"],
            "title": result["title"],
            "content": result["content"],
            "score": result["score"],
        }
        for result in valid_results
    ]

    logger.debug(
        f"Formatted {len(formatted_results)} results with url, title, content, and score"
    )
    return formatted_results


if __name__ == "__main__":
    # Check if SEARXNG_URL is set
    if not SEARCH_URL:
        print("Error: SEARXNG_URL environment variable is not set.")
        print("Please set it before running the demo.")
        exit(1)

    # Demonstrate search_web
    print("Demonstrating search_web function...")

    # Simple search
    query = "Python programming language"
    print(f"\nSearching for: '{query}'")
    results = search_web(query, count=3)

    print(f"\nFound {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print(f"Title: {result['title']}")
        print(f"URL: {result['url']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Content snippet: {result['content'][:200]}...")

    # Example with config
    print("\n" + "=" * 50)
    print("Demo with config (include sites and min_date):")
    try:
        config = {"include_sites": ["python.org"], "min_date": datetime(2024, 1, 1)}
        results_config = search_web("Python 3.12 features", count=2, config=config)
        print(f"Found {len(results_config)} results with config:")
        for i, result in enumerate(results_config, 1):
            print(f"\n--- Config Result {i} ---")
            print(f"Title: {result['title']}")
            print(f"URL: {result['url']}")
    except Exception as e:
        print(f"Config demo failed (expected if no results or env issues): {e}")
