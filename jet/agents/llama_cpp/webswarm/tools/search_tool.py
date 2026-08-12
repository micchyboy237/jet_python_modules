import httpx
from config import AtomConfig
from langchain_core.tools import tool
from tenacity import retry, stop_after_attempt, wait_exponential


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
async def _serper_search(query: str, api_key: str, num_results: int) -> list[dict]:
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            json={"q": query, "num": num_results},
        )
        resp.raise_for_status()
        data = resp.json()
    results = []
    for item in data.get("organic", [])[:num_results]:
        results.append(
            {
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": item.get("snippet", ""),
            }
        )
    return results


def create_search_tool(config: AtomConfig):
    @tool
    async def search(query: str) -> list[dict]:
        """Search the web using Serper API. Returns top-N results with title, url, snippet."""
        return await _serper_search(
            query, config.serper_api_key, config.max_search_results
        )

    return search
