import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from jet.libs.smolagents.utils.debug_saver import DebugSaver
from jet.utils.inspect_utils import get_entry_file_dir, get_entry_file_name
from smolagents import Tool

logger = logging.getLogger(__name__)


@dataclass
class SearXNGSearchTool(Tool):
    """
    Web search tool that performs searches using the SearXNG search engine.
    Args:
        instance_url: URL of the SearXNG instance
        max_results: Maximum number of search results to return (default: 10)
        rate_limit: Maximum queries per second (None = no limit)
        verbose: Enable detailed logging (default: True)
        logs_dir: Directory to save structured call logs + full markdown results (default: None)
        timeout: HTTP request timeout in seconds
    """

    name = "web_search"
    description = (
        "Performs a web search using SearXNG (privacy-friendly metasearch engine) "
        "and returns the top results as formatted markdown with titles, URLs, and snippets."
    )
    inputs = {
        "query": {"type": "string", "description": "The search query to perform."}
    }
    output_type = "string"

    def __init__(
        self,
        instance_url: str = "http://jethros-macbook-air.local:8888",
        max_results: int = 10,
        rate_limit: float | None = 2.0,
        timeout: int = 10,
        verbose: bool = True,
        logs_dir: str | Path | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.instance_url = instance_url.rstrip("/")
        self.max_results = max_results
        self.rate_limit = rate_limit
        self.timeout = timeout
        self.verbose = verbose

        _caller_base_dir = (
            Path(get_entry_file_dir())
            / "generated"
            / Path(get_entry_file_name()).stem
            / "searxng_search_tool_logs"
        )
        base_dir = Path(logs_dir).resolve() if logs_dir else _caller_base_dir

        self.debug_saver = DebugSaver(
            tool_name=self.name,
            base_dir=base_dir,
        )

        self._min_interval = 1.0 / rate_limit if rate_limit else 0.0
        self._last_request_time = 0.0

        if self.verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.WARNING)

    def _enforce_rate_limit(self) -> None:
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
        Saves:
          - request.json
          - full_results.md
          - response.json (summary)
          - error.txt (if failed)
        """
        request_data = {
            "query": query,
            "max_results": self.max_results,
            "instance_url": self.instance_url,
            "timeout": self.timeout,
            "rate_limit_qps": self.rate_limit,
        }

        self.debug_saver.save_json("request.json", request_data, indent=2)
        logger.info("Saved request.json")

        with self.debug_saver.new_call(request_data) as call_dir:
            if self.verbose:
                logger.debug(f"Query → {query!r}")
                logger.debug(f"Target instance: {self.instance_url}")

            self._enforce_rate_limit()

            try:
                params = {
                    "q": query,
                    "format": "json",
                    "pageno": 1,
                    "categories": "general",
                }

                if self.verbose:
                    logger.debug(f"GET {self.instance_url}/search | params={params}")

                response = requests.get(
                    f"{self.instance_url}/search",
                    params=params,
                    timeout=self.timeout,
                    headers={
                        "User-Agent": "smolagents/SearXNGSearchTool[](https://github.com)"
                    },
                )
                response.raise_for_status()

                data = response.json()
                results: list[dict[str, Any]] = data.get("results", [])

                if not results:
                    msg = f"No results found for query: {query!r}"
                    if self.verbose:
                        logger.warning(msg)
                    self.debug_saver.save("full_results.md", msg)
                    return msg

                top_results = results[: self.max_results]

                formatted_lines = []
                for idx, result in enumerate(top_results, 1):
                    title = result.get("title", "No title").strip()
                    url = result.get("url", "").strip()
                    content = result.get("content", "").strip()

                    formatted_lines.append(f"**{idx}. [{title}]({url})**")

                    if content:
                        if len(content) > 500:
                            content = content[:497] + "..."
                        formatted_lines.append("")
                        formatted_lines.append(content)
                    formatted_lines.append("")

                result_text = "\n".join(formatted_lines).strip()

                self.debug_saver.save("full_results.md", result_text, encoding="utf-8")

                self.debug_saver.save_json(
                    "response.json",
                    {
                        "result_count": len(top_results),
                        "formatted_length": len(result_text),
                        "preview": result_text[:600] + "..."
                        if len(result_text) > 600
                        else result_text,
                        "full_markdown_file": "full_results.md",
                    },
                    indent=2,
                    ensure_ascii=False,
                )

                if self.verbose:
                    logger.debug(
                        f"Saved full markdown → {call_dir / 'full_results.md'}"
                    )
                    logger.debug(
                        f"Returning {len(top_results)} results ({len(result_text):,} chars)"
                    )

                return result_text

            except requests.exceptions.RequestException as e:
                err_msg = f"Error performing SearXNG search: {str(e)}\nTry another instance or check your network."
                if self.verbose:
                    logger.error(err_msg)
                self.debug_saver.save("error.txt", str(e))
                self.debug_saver.save("full_results.md", err_msg)
                return err_msg

            except Exception as e:
                err_msg = f"Unexpected error during SearXNG search: {str(e)}"
                if self.verbose:
                    logger.exception(err_msg)
                self.debug_saver.save("error.txt", str(e))
                self.debug_saver.save("full_results.md", err_msg)
                return err_msg

    @classmethod
    def with_instance(cls, instance_url: str, **kwargs):
        return cls(instance_url=instance_url, **kwargs)
