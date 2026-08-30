from abc import ABC, abstractmethod
from typing import List

from models import SearchResult


class SearchProvider(ABC):
    @abstractmethod
    async def search(self, query: str, num_results: int = 20) -> List[SearchResult]: ...


class SerpAPISearchProvider(SearchProvider):
    """Placeholder. Replace with actual SerpAPI/Tavily/Bing integration."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    async def search(self, query: str, num_results: int = 20) -> List[SearchResult]:
        # TODO: Implement async HTTP call to search API
        # Return list of SearchResult objects sorted by relevance score
        raise NotImplementedError("Implement SerpAPI search integration")
