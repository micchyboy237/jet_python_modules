from agent.config import Config
from jet.search.searxng import search_searxng

SEARCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web using SearXNG with caching, deduplication, and relevance scoring.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Optimized search query"},
                "num_results": {
                    "type": "integer",
                    "description": "Max results (default 5)",
                    "default": 5,
                },
                "include_sites": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Limit to specific domains",
                },
                "exclude_sites": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Exclude specific domains",
                },
            },
            "required": ["query"],
        },
    },
}


def web_search(
    query: str,
    num_results: int = 5,
    include_sites: list[str] | None = None,
    exclude_sites: list[str] | None = None,
) -> str:
    try:
        results = search_searxng(
            query=query,
            query_url=Config.SEARXNG_URL,
            count=num_results,
            min_score=0.1,
            use_cache=True,
            include_sites=include_sites,
            exclude_sites=exclude_sites,
            max_retries=2,
        )
    except Exception as e:
        return f"Search error: {type(e).__name__}: {str(e)[:300]}"

    if not results:
        return (
            "No relevant search results found. Try rephrasing or broadening the query."
        )

    formatted = []
    for r in results:
        date = r.get("publishedDate", "N/A")
        score = r.get("score", "N/A")
        formatted.append(
            f"**{r['title']}**\n"
            f"URL: {r['url']}\n"
            f"Snippet: {r.get('content', '')[:300]}\n"
            f"Date: {date} | Relevance: {score}"
        )
    return "\n\n---\n\n".join(formatted)
