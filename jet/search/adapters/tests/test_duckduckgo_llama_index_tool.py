import pytest
from typing import List, Dict
from unittest.mock import patch
from duckduckgo_search import DDGS
from jet.search.adapters.duckduckgo_llama_index_tool import DuckDuckGoSearchToolSpec


@pytest.fixture
def search_tool():
    return DuckDuckGoSearchToolSpec()


class TestDuckDuckGoInstantSearch:
    def test_instant_search_success(self, search_tool):
        # Given: A valid query for instant search
        query = "python programming"
        # Mocked response
        expected = [{"text": "Python is a programming language", "url": None}]

        # When: Performing an instant search
        with patch.object(DDGS, "answers", return_value=expected):
            result = search_tool.duckduckgo_instant_search(query)

        # Then: The result should match the expected output
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_instant_search_empty_result(self, search_tool):
        # Given: A query with no instant answers
        query = "nonexistent topic 123"
        expected: List[Dict] = []

        # When: Performing an instant search
        with patch.object(DDGS, "answers", return_value=expected):
            result = search_tool.duckduckgo_instant_search(query)

        # Then: The result should be an empty list
        assert result == expected, f"Expected empty list, but got {result}"


class TestDuckDuckGoFullSearch:
    def test_full_search_success(self, search_tool):
        # Given: A valid query for full search with specific region and max results
        query = "python programming"
        region = "us-en"
        max_results = 2
        expected = [
            {"title": "Python Official Site", "href": "https://python.org",
                "body": "Official Python site"},
            {"title": "Python Docs", "href": "https://docs.python.org",
                "body": "Python documentation"}
        ]

        # When: Performing a full search
        with patch.object(DDGS, "text", return_value=expected):
            result = search_tool.duckduckgo_full_search(
                query, region=region, max_results=max_results)

        # Then: The result should match the expected output
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_full_search_invalid_region(self, search_tool):
        # Given: A query with an invalid region
        query = "python programming"
        region = "invalid-region"
        max_results = 1
        expected = [{"title": "Python Guide",
                     "href": "https://python.org", "body": "Guide to Python"}]

        # When: Performing a full search with invalid region
        with patch.object(DDGS, "text", return_value=expected):
            result = search_tool.duckduckgo_full_search(
                query, region=region, max_results=max_results)

        # Then: The result should still return valid results (API may ignore invalid region)
        assert result == expected, f"Expected {expected}, but got {result}"


@pytest.fixture(autouse=True)
def cleanup():
    # Clean up any resources if needed
    yield
