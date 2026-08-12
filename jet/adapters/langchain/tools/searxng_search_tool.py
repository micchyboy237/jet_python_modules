"""Tool for the SearXNG search API using jet.search.searxng core."""

import json
import os
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

from jet.logger import logger
from jet.search.searxng import SearchResult, search_searxng
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class SearXNGInput(BaseModel):
    """Input schema for the SearXNG search tool."""

    query: str = Field(description="The search query to look up.")
    count: Optional[int] = Field(
        default=None,
        description="Maximum number of results to return. If None, returns all filtered results.",
    )
    min_score: float = Field(
        default=0.1, description="Minimum relevance score threshold (0.0 to 1.0)."
    )
    engines: Optional[List[str]] = Field(
        default=None,
        description="Specific search engines to use (e.g., ['google', 'bing']).",
    )
    categories: List[str] = Field(
        default=["general"],
        description="Search categories (e.g., ['general', 'news', 'science']).",
    )
    language: str = Field(default="en", description="Search language code.")
    safesearch: int = Field(
        default=0, description="Safe search level: 0 (off), 1 (moderate), 2 (strict)."
    )
    years_ago: int = Field(
        default=1,
        description="Limit results to this many years ago when no explicit min_date is set.",
    )


class SearXNGSearchResults(BaseTool):
    """SearXNG search tool that leverages jet.search.searxng core functionality.

    This tool provides privacy-respecting metasearch capabilities with built-in
    caching, result filtering, deduplication, and scoring.

    Setup:
        Ensure SEARXNG_URL environment variable is set or pass query_url during init.
        Redis must be available for caching.

    Instantiation:
        .. code-block:: python
            from jet.adapters.langchain.searxng_search_tool import SearXNGSearchResults

            tool = SearXNGSearchResults(
                max_results=5,
                engines=["google", "duckduckgo"],
                categories=["general"]
            )

    Invocation:
        .. code-block:: python
            tool.invoke({"query": "latest AI developments", "count": 3})
    """

    name: str = "searxng_search"
    description: str = (
        "A privacy-respecting metasearch engine wrapper around SearXNG. "
        "Useful for answering questions about current events, research, and general knowledge. "
        "Supports filtering by engine, category, date, and relevance score. "
        "Input should be a search query string."
    )
    args_schema: Type[BaseModel] = SearXNGInput

    # Tool configuration defaults
    default_count: Optional[int] = Field(default=5, alias="max_results")
    default_min_score: float = Field(default=0.1)
    default_engines: Optional[List[str]] = Field(default=None)
    default_categories: List[str] = Field(default=["general"])
    default_language: str = Field(default="en")
    default_safesearch: int = Field(default=0)
    default_years_ago: int = Field(default=1)
    query_url: Optional[str] = Field(
        default=os.getenv("SEARXNG_URL", "http://localhost:8888"),
        description="Custom SearXNG instance URL.",
    )

    # Output formatting
    output_format: Literal["string", "json", "list"] = Field(
        default="string",
        description="Output format: 'string' (concatenated text), 'json' (JSON string), or 'list' (raw dicts).",
    )
    keys_to_include: Optional[List[str]] = Field(
        default=["title", "url", "content", "score", "engine"],
        description="Which keys from each result to include in output. None includes all.",
    )
    results_separator: str = Field(
        default="\n\n", description="Separator between results in string format."
    )

    response_format: Literal["content_and_artifact"] = "content_and_artifact"

    def _format_results(self, results: List[SearchResult]) -> Union[str, List[Dict]]:
        """Format search results according to output_format specification."""
        if not results:
            return "No results found." if self.output_format == "string" else []

        # Filter keys if specified
        filtered = [
            {
                k: v
                for k, v in r.items()
                if not self.keys_to_include or k in self.keys_to_include
            }
            for r in results
        ]

        if self.output_format == "list":
            return filtered
        elif self.output_format == "json":
            return json.dumps(filtered, indent=2, ensure_ascii=False)
        else:  # string
            parts = []
            for item in filtered:
                lines = [f"{k}: {v}" for k, v in item.items() if v is not None]
                parts.append("\n".join(lines))
            return self.results_separator.join(parts)

    def _run(
        self,
        query: str,
        count: Optional[int] = None,
        min_score: Optional[float] = None,
        engines: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        language: Optional[str] = None,
        safesearch: Optional[int] = None,
        years_ago: Optional[int] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[Union[str, List[Dict]], List[SearchResult]]:
        """Execute SearXNG search with parameter merging and logging."""
        # Merge invocation params with tool defaults
        effective_count = count if count is not None else self.default_count
        effective_min_score = (
            min_score if min_score is not None else self.default_min_score
        )
        effective_engines = engines if engines is not None else self.default_engines
        effective_categories = (
            categories if categories is not None else self.default_categories
        )
        effective_language = language or self.default_language
        effective_safesearch = (
            safesearch if safesearch is not None else self.default_safesearch
        )
        effective_years_ago = (
            years_ago if years_ago is not None else self.default_years_ago
        )

        logger.info(
            f"SearXNGSearchResults: Searching for '{query}' | "
            f"count={effective_count}, engines={effective_engines}, "
            f"categories={effective_categories}"
        )

        logger.debug(f"SearXNGSearchResults._run called with:")
        logger.debug(f"  query: {query}")
        logger.debug(f"  count: {effective_count}")
        logger.debug(
            f"  engines: {effective_engines} (type: {type(effective_engines).__name__})"
        )
        logger.debug(
            f"  categories: {effective_categories} (type: {type(effective_categories).__name__})"
        )
        logger.debug(f"  language: {effective_language}")
        logger.debug(f"  safesearch: {effective_safesearch}")
        logger.debug(f"  years_ago: {effective_years_ago}")

        try:
            results = search_searxng(
                query=query,
                query_url=self.query_url,
                count=effective_count,
                min_score=effective_min_score,
                engines=effective_engines,
                language=effective_language,
                categories=effective_categories,
                safesearch=effective_safesearch,
                years_ago=effective_years_ago,
                use_cache=True,  # Core function handles its own Redis cache
            )

            formatted = self._format_results(results)

            logger.success(
                f"SearXNGSearchResults: Returned {len(results)} results for '{query}'"
            )
            return formatted, results

        except Exception as e:
            logger.error(f"SearXNGSearchResults: Search failed for '{query}': {e}")
            error_msg = f"Search error: {repr(e)}"
            return error_msg, []
