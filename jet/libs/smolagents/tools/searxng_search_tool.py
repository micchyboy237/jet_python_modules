import requests
import time
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from smolagents import Tool

@dataclass
class SearXNGSearchTool(Tool):
    """
    Web search tool that uses SearXNG (open-source metasearch engine).
    Completely free, no API key required.
    
    Args:
        instance_url: URL of the SearXNG instance (default: a fast public one)
        max_results: Maximum number of search results to return (default: 10)
        rate_limit: Maximum queries per second (set to None for no limit)
    """
    name = "searxng_search"
    description = (
        "Performs a web search using SearXNG (privacy-friendly metasearch engine) "
        "and returns the top results as formatted markdown with titles, URLs, and snippets."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "The search query to perform."
        }
    }
    output_type = "string"

    def __init__(
        self,
        instance_url: str = "http://jethros-macbook-air.local:8888", # "https://searx.tiekoetter.net",
        max_results: int = 10,
        rate_limit: Optional[float] = 2.0,  # 2 queries per second by default
        timeout: int = 10,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.instance_url = instance_url.rstrip("/")  # normalize
        self.max_results = max_results
        self.rate_limit = rate_limit
        self.timeout = timeout

        self._min_interval = 1.0 / rate_limit if rate_limit else 0.0
        self._last_request_time = 0.0

    def _enforce_rate_limit(self) -> None:
        """Sleep if needed to respect rate limit"""
        if not self.rate_limit:
            return
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()

    def forward(self, query: str) -> str:
        """
        Execute search and return formatted markdown results.
        """
        self._enforce_rate_limit()

        try:
            params = {
                "q": query,
                "format": "json",
                "pageno": 1,
                "categories": "general",  # you can also use: images, videos, news, etc.
            }

            response = requests.get(
                f"{self.instance_url}/search",
                params=params,
                timeout=self.timeout,
                headers={"User-Agent": "smolagents/SearXNGSearchTool[](https://github.com)"}
            )
            response.raise_for_status()

            data = response.json()
            results: List[Dict[str, Any]] = data.get("results", [])

            if not results:
                return f"No results found for query: '{query}'"

            # Take only the top N results
            top_results = results[:self.max_results]

            formatted = ["## Search Results via SearXNG\n"]
            for idx, result in enumerate(top_results, 1):
                title = result.get("title", "No title")
                url = result.get("url", "#")
                content = result.get("content", "").strip()

                # Clean up snippet
                if len(content) > 300:
                    content = content[:297] + "..."

                formatted.append(
                    f"{idx}. **[{title}]({url})**\n"
                    f"   {content}\n"
                )

            return "\n".join(formatted)

        except requests.exceptions.RequestException as e:
            return f"Error performing SearXNG search: {str(e)}\nTry another instance or check your network."
        except Exception as e:
            return f"Unexpected error during SearXNG search: {str(e)}"

    @classmethod
    def with_instance(cls, instance_url: str, **kwargs):
        """Convenience factory method"""
        return cls(instance_url=instance_url, **kwargs)