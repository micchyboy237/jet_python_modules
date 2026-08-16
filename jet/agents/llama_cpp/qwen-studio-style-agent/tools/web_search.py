import httpx
from agent.config import Config
from tenacity import retry, stop_after_attempt, wait_exponential

SEARCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for current information. Returns titles, URLs, snippets, and dates.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Optimized search query"},
                "num_results": {
                    "type": "integer",
                    "description": "Number of results (default 5)",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
}


@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=5))
def web_search(query: str, num_results: int = 5) -> str:
    num_results = min(max(1, num_results), 10)

    resp = httpx.get(
        f"{Config.SEARXNG_BASE_URL}/search",
        params={"q": query, "format": "json", "engines": "google,bing,duckduckgo"},
        timeout=15.0,
    )
    resp.raise_for_status()
    results = resp.json().get("results", [])[:num_results]

    if not results:
        return "No search results found. Try rephrasing the query."

    formatted = []
    for r in results:
        date = r.get("publishedDate", "N/A")
        formatted.append(
            f"**{r['title']}**\nURL: {r['url']}\nSnippet: {r.get('content', '')[:300]}\nDate: {date}"
        )
    return "\n\n---\n\n".join(formatted)
