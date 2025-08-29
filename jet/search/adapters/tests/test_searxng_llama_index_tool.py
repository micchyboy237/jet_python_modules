import pytest
from typing import List, Dict
from datetime import datetime
from unittest.mock import patch, Mock
from jet.search.searxng import SearchResult, QueryResponse
from jet.search.adapters.searxng_llama_index_tool import SearxngSearchToolSpec
from jet.cache.redis import RedisCache


@pytest.fixture
def search_tool():
    return SearxngSearchToolSpec(base_url="https://searxng.example.com", redis_config={"port": 3101})


class TestSearxngInstantSearch:
    def test_instant_search_success(self, search_tool):
        # Given: A valid query for instant search
        query = "python programming"
        expected = [{"text": "Python is a high-level programming language"}]
        mock_response = QueryResponse(
            query=query, answers=["Python is a high-level programming language"])

        # When: Performing an instant search
        with patch("jet.search.adapters.searxng_llama_index_tool.fetch_search_results", return_value=mock_response):
            result = search_tool.searxng_instant_search(query)

        # Then: The result should match the expected output
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_instant_search_no_results(self, search_tool):
        # Given: A query with no instant answers
        query = "nonexistent topic 123"
        expected: List[Dict] = []
        mock_response = QueryResponse(query=query, answers=[])

        # When: Performing an instant search
        with patch("jet.search.adapters.searxng_llama_index_tool.fetch_search_results", return_value=mock_response):
            result = search_tool.searxng_instant_search(query)

        # Then: The result should be an empty list
        assert result == expected, f"Expected empty list, but got {result}"


class TestSearxngFullSearch:
    def test_full_search_success(self, search_tool):
        # Given: A valid query for full search with specific parameters
        query = "python programming"
        count = 2
        expected = [
            SearchResult(
                id="key1",
                url="https://python.org",
                title="Python Official Site",
                content="Official Python site",
                engine="google",
                template="default",
                parsed_url=["python.org"],
                engines=["google"],
                positions=[1],
                publishedDate="2025-01-01T00:00:00",
                score=0.9,
                category="general"
            ),
            SearchResult(
                id="key2",
                url="https://docs.python.org",
                title="Python Docs",
                content="Python documentation",
                engine="bing",
                template="default",
                parsed_url=["docs.python.org"],
                engines=["bing"],
                positions=[2],
                publishedDate="2025-01-02T00:00:00",
                score=0.8,
                category="general"
            )
        ]
        mock_response = QueryResponse(
            query=query, results=expected, number_of_results=2)

        # When: Performing a full search
        with patch("jet.search.adapters.searxng_llama_index_tool.fetch_search_results", return_value=mock_response), \
                patch.object(RedisCache, "get", return_value=None), \
                patch.object(RedisCache, "set") as mock_cache_set:
            result = search_tool.searxng_full_search(
                query, count=count, min_score=0.1)

        # Then: The result should match the expected output
        assert result == expected, f"Expected {expected}, but got {result}"
        mock_cache_set.assert_called_once()

    def test_full_search_with_cache(self, search_tool):
        # Given: A query with cached results
        query = "python programming"
        count = 1
        expected = [
            SearchResult(
                id="key1",
                url="https://python.org",
                title="Python Official Site",
                content="Official Python site",
                engine="google",
                template="default",
                parsed_url=["python.org"],
                engines=["google"],
                positions=[1],
                publishedDate="2025-01-01T00:00:00",
                score=0.9,
                category="general"
            )
        ]
        cached_response = {"results": expected, "number_of_results": 1}

        # When: Performing a full search with cache hit
        with patch("jet.search.adapters.searxng_llama_index_tool.fetch_search_results", side_effect=Exception("Should not be called")), \
                patch.object(RedisCache, "get", return_value=cached_response):
            result = search_tool.searxng_full_search(
                query, count=count, min_score=0.1)

        # Then: The result should match the cached output
        assert result == expected, f"Expected {expected}, but got {result}"


@pytest.fixture(autouse=True)
def cleanup():
    # Clean up any resources if needed
    yield
