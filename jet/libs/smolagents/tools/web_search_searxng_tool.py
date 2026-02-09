import requests
from smolagents import Tool


class WebSearchTool(Tool):
    name = "web_search"
    description = (
        "Performs a web search using a SearXNG instance and returns "
        "markdown-formatted search results with titles, links, and descriptions."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "The search query to perform.",
        }
    }
    output_type = "string"

    def __init__(
        self,
        max_results: int = 10,
        searxng_base_url: str = "http://searxng.local:8888",
        language: str = "en",
    ):
        super().__init__()
        self.max_results = max_results
        self.searxng_base_url = searxng_base_url.rstrip("/")
        self.language = language

    def forward(self, query: str) -> str:
        results = self.search(query)
        if not results:
            raise Exception("No results found! Try a less restrictive/shorter query.")
        return self.parse_results(results)

    def search(self, query: str) -> list[dict[str, str]]:
        return self.search_searxng(query)

    def parse_results(self, results: list[dict[str, str]]) -> str:
        return "## Search Results\n\n" + "\n\n".join(
            f"[{result['title']}]({result['link']})\n{result['description']}"
            for result in results
        )

    def search_searxng(self, query: str) -> list[dict[str, str]]:
        """
        Uses the SearXNG JSON API:
        https://docs.searxng.org/dev/search_api.html
        """
        response = requests.get(
            f"{self.searxng_base_url}/search",
            params={
                "q": query,
                "format": "json",
                "language": self.language,
                "categories": "general",
            },
            headers={
                "User-Agent": "Mozilla/5.0",
            },
            timeout=10,
        )
        response.raise_for_status()

        data = response.json()
        raw_results = data.get("results", [])

        results: list[dict[str, str]] = []
        for item in raw_results[: self.max_results]:
            title = item.get("title")
            link = item.get("url")
            description = item.get("content") or ""

            if not title or not link:
                continue

            results.append(
                {
                    "title": title.strip(),
                    "link": link.strip(),
                    "description": description.strip(),
                }
            )

        return results
