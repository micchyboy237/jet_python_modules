import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from jet.adapters.llama_cpp.types import LLAMACPP_EMBED_KEYS
from jet.libs.smolagents.utils import DebugSaver
from jet.utils.inspect_utils import get_entry_file_dir, get_entry_file_name
from jet.vectors.semantic_search.web_search import hybrid_search
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
        instance_url: str = "http://searxng.local:8888",
        max_results: int | None = 10,
        rate_limit: float | None = 2.0,
        timeout: int = 10,
        verbose: bool = True,
        logs_dir: str | Path | None = None,
        embed_model: LLAMACPP_EMBED_KEYS = "nomic-embed-text",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.instance_url = instance_url.rstrip("/")
        self.max_results = max_results
        self.rate_limit = rate_limit
        self.timeout = timeout
        self.verbose = verbose
        self.embed_model = embed_model

        _caller_base_dir = (
            Path(get_entry_file_dir())
            / "generated"
            / Path(get_entry_file_name()).stem
            / "searxng_search_tool_logs"
        )
        self.base_dir = Path(logs_dir).resolve() if logs_dir else _caller_base_dir

        self.debug_saver = DebugSaver(
            tool_name=self.name,
            base_dir=self.base_dir,
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
        result = asyncio.run(
            hybrid_search(
                query,
                llm_log_dir=self.base_dir,
                use_cache=True,
                early_stop=True,
            )
        )

        return result["llm_response"]

    @classmethod
    def with_instance(cls, instance_url: str, **kwargs):
        return cls(instance_url=instance_url, **kwargs)
