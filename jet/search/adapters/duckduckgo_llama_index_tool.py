import importlib.util
from typing import List, Dict, Optional
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from jet.logger import logger


class DuckDuckGoSearchToolSpec(BaseToolSpec):
    """DuckDuckGoSearch tool spec with rate limit handling."""

    spec_functions = ["duckduckgo_instant_search", "duckduckgo_full_search"]

    def __init__(self) -> None:
        if not importlib.util.find_spec("duckduckgo_search"):
            raise ImportError(
                "DuckDuckGoSearchToolSpec requires the duckduckgo_search package to be installed."
            )
        super().__init__()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(Exception),
        before_sleep=lambda retry_state: logger.info(
            f"Rate limit hit, retrying after {retry_state.next_action.sleep} seconds..."
        )
    )
    def duckduckgo_instant_search(self, query: str) -> List[Dict]:
        """
        Make a query to DuckDuckGo API to receive an instant answer.

        Args:
            query (str): The query to be passed to DuckDuckGo.

        Returns:
            List[Dict]: List of instant answer results.

        Raises:
            Exception: If the API call fails after retries.
        """
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddg:
                logger.info(f"Executing instant search for query: {query}")
                return list(ddg.answers(query))
        except Exception as e:
            logger.error(
                f"Instant search failed for query '{query}': {str(e)}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(Exception),
        before_sleep=lambda retry_state: logger.info(
            f"Rate limit hit, retrying after {retry_state.next_action.sleep} seconds..."
        )
    )
    def duckduckgo_full_search(
        self,
        query: str,
        region: Optional[str] = "wt-wt",
        max_results: Optional[int] = 10,
    ) -> List[Dict]:
        """
        Make a query to DuckDuckGo search to receive full search results.

        Args:
            query (str): The query to be passed to DuckDuckGo.
            region (Optional[str]): The region to be used for the search in [country-language] convention, ex us-en, uk-en, ru-ru, etc...
            max_results (Optional[int]): The maximum number of results to be returned.

        Returns:
            List[Dict]: List of search results.

        Raises:
            Exception: If the API call fails after retries.
        """
        try:
            from duckduckgo_search import DDGS
            params = {
                "keywords": query,
                "region": region,
                "max_results": int(max_results) if max_results else None,
            }
            with DDGS() as ddg:
                logger.info(f"Executing full search with params: {params}")
                return list(ddg.text(**params))
        except Exception as e:
            logger.error(f"Full search failed for query '{query}': {str(e)}")
            raise
