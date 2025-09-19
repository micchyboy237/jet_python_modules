from typing import Optional, List, Dict, Any, TypedDict
from datetime import datetime
from jet.search.searxng import search_searxng
from jet.logger import logger

SEARCH_URL = "http://jethros-macbook-air.local:3000"

class WebSearchResult(TypedDict):
    url: str
    title: str
    content: str
    score: float

def search_web(
    query: str,
    count: Optional[int] = 10,
    config: Optional[Dict[str, Any]] = None
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
        result for result in results
        if all(key in result for key in required_keys)
    ]

    logger.debug(f"Filtered to {len(valid_results)} results with all required keys (url, title, content, score)")

    # Format valid results to include only url, title, content, and score
    formatted_results: List[WebSearchResult] = [
        {
            "url": result["url"],
            "title": result["title"],
            "content": result["content"],
            "score": result["score"]
        }
        for result in valid_results
    ]

    logger.debug(f"Formatted {len(formatted_results)} results with url, title, content, and score")
    return formatted_results
