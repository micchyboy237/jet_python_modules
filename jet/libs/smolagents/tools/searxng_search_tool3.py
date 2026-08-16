"""
searxng_search_tool.py

A smolagents `Tool` that performs web search using a self-hosted (or remote)
SearXNG instance's JSON API.

Requires the SearXNG instance to have JSON output enabled, e.g. in settings.yml:

    search:
      formats:
        - html
        - json

Env var:
    SEARXNG_URL   Base URL of the SearXNG instance. Defaults to "http://localhost:8888".

Example:
    ```python
    from searxng_search_tool import SearXNGSearchTool

    tool = SearXNGSearchTool(max_results=5)
    print(tool("smolagents github"))
    ```
"""

import logging
import os
import time
from typing import Any

from smolagents.tools import Tool

logger = logging.getLogger("searxng_search_tool")
if not logger.handlers:
    # Only attach a default handler if the host app hasn't configured logging itself.
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s")
    )
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


class SearXNGSearchTool(Tool):
    """Web search tool that queries a SearXNG metasearch instance and returns
    results formatted as markdown (title, link, snippet).

    Args:
        instance_url (`str`, *optional*): Base URL of the SearXNG instance.
            Defaults to the `SEARXNG_URL` env var, or "http://localhost:8888" if unset.
        max_results (`int`, default `10`): Maximum number of results to return.
        categories (`str`, *optional*): Comma-separated SearXNG categories
            (e.g. "general", "news", "images"). Defaults to None (SearXNG's default).
        language (`str`, default `"en"`): Language code for results (e.g. "en", "fr", or "all").
        safesearch (`int`, default `0`): Safe search level: 0 (off), 1 (moderate), 2 (strict).
        time_range (`str`, *optional*): One of "day", "month", "year" if the engine supports it.
        rate_limit (`float`, default `1.0`): Max queries per second. Set to `None` to disable.
        timeout (`int`, default `10`): Request timeout in seconds.

    Example:
        ```python
        >>> from searxng_search_tool import SearXNGSearchTool
        >>> web_search_tool = SearXNGSearchTool(max_results=5)
        >>> results = web_search_tool("Hugging Face")
        >>> print(results)
        ```
    """

    name = "web_search"
    description = (
        "Performs a web search using a self-hosted SearXNG metasearch engine and returns "
        "a string of the top search results formatted as markdown with titles, links, and snippets."
    )
    inputs = {
        "query": {"type": "string", "description": "The search query to perform."},
    }
    output_type = "string"

    def __init__(
        self,
        instance_url: str | None = None,
        max_results: int = 10,
        categories: str | None = None,
        language: str = "en",
        safesearch: int = 0,
        time_range: str | None = None,
        rate_limit: float | None = 1.0,
        timeout: int = 10,
        **kwargs,
    ):
        super().__init__()

        self.instance_url = (
            instance_url or os.getenv("SEARXNG_URL", "http://localhost:8888")
        ).rstrip("/")
        self.max_results = max_results
        self.categories = categories
        self.language = language
        self.safesearch = safesearch
        self.time_range = time_range
        self.timeout = timeout

        self.rate_limit = rate_limit
        self._min_interval = 1.0 / rate_limit if rate_limit else 0.0
        self._last_request_time = 0.0

        try:
            import requests  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "You must install package `requests` to run this tool: for instance run `pip install requests`."
            ) from e

        logger.info(
            "SearXNGSearchTool initialized (instance_url=%s, max_results=%s, categories=%s, language=%s)",
            self.instance_url,
            self.max_results,
            self.categories,
            self.language,
        )

    def _enforce_rate_limit(self) -> None:
        if not self.rate_limit:
            return
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._min_interval:
            wait_time = self._min_interval - elapsed
            logger.debug("Rate limit active, sleeping for %.3fs", wait_time)
            time.sleep(wait_time)
        self._last_request_time = time.time()

    def _build_params(self, query: str) -> dict[str, Any]:
        params: dict[str, Any] = {
            "q": query,
            "format": "json",
            "language": self.language,
            "safesearch": self.safesearch,
        }
        if self.categories:
            params["categories"] = self.categories
        if self.time_range:
            params["time_range"] = self.time_range
        return params

    def forward(self, query: str) -> str:
        import requests

        self._enforce_rate_limit()

        url = f"{self.instance_url}/search"
        params = self._build_params(query)

        logger.info("Querying SearXNG: url=%s query=%r", url, query)

        try:
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
        except requests.exceptions.Timeout as e:
            logger.error("SearXNG request timed out after %ss: %s", self.timeout, e)
            raise Exception(
                f"SearXNG request timed out after {self.timeout}s. Try again or check the instance."
            ) from e
        except requests.exceptions.ConnectionError as e:
            logger.error(
                "Could not connect to SearXNG instance at %s: %s", self.instance_url, e
            )
            raise Exception(
                f"Could not connect to SearXNG instance at {self.instance_url}. "
                "Check that it's running and SEARXNG_URL is correct."
            ) from e
        except requests.exceptions.HTTPError as e:
            logger.error("SearXNG returned an HTTP error: %s", e)
            if response.status_code == 403:
                raise Exception(
                    "SearXNG returned 403 Forbidden. This usually means JSON format is not enabled "
                    "on the instance (add 'json' under search.formats in settings.yml)."
                ) from e
            raise Exception(f"SearXNG returned an HTTP error: {e}") from e

        try:
            data = response.json()
        except ValueError as e:
            logger.error("Failed to parse SearXNG JSON response: %s", e)
            raise Exception(
                "SearXNG did not return valid JSON. Confirm the JSON format is enabled."
            ) from e

        results = data.get("results", [])
        logger.info(
            "SearXNG returned %d raw result(s) for query=%r", len(results), query
        )

        if not results:
            raise Exception("No results found! Try a less restrictive/shorter query.")

        results = results[: self.max_results]
        formatted = self._format_results(results)

        logger.info(
            "Returning %d formatted result(s) for query=%r", len(results), query
        )
        return formatted

    def _format_results(self, results: list) -> str:
        entries = []
        for result in results:
            title = result.get("title", "Untitled")
            link = result.get("url", "")
            content = result.get("content", "").strip()
            engine = result.get("engine", "")
            snippet = f"[{title}]({link})"
            if engine:
                snippet += f"\nSource engine: {engine}"
            if content:
                snippet += f"\n{content}"
            entries.append(snippet)
        return "## Search Results\n\n" + "\n\n".join(entries)


__all__ = ["SearXNGSearchTool"]
