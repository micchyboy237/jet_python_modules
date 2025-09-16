import httpx
import os
import json
import asyncio
from dotenv import load_dotenv
from typing import Dict, List, Literal, Optional, Tuple, TypedDict, Union
from langchain_community.tools.tavily_search import TavilySearchResults, TavilyAnswer
from langchain_core.messages import ToolMessage
from langchain_core.callbacks import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from pydantic import BaseModel

# Typed dictionaries for structured inputs and outputs


class SearchConfig(TypedDict):
    """Configuration for Tavily search."""
    max_results: int
    search_depth: Literal["basic", "advanced"]
    include_answer: bool
    include_raw_content: bool
    include_images: bool
    include_domains: Optional[List[str]]
    exclude_domains: Optional[List[str]]
    redis_host: str
    redis_port: int


class SearchResult(TypedDict):
    """Structure for a single search result."""
    title: str
    url: str
    content: str
    score: Optional[float]
    raw_content: Optional[str]


class SearchArtifact(TypedDict):
    """Structure for the full search artifact."""
    query: str
    results: List[SearchResult]
    answer: Optional[str]
    images: Optional[List[str]]
    response_time: Optional[float]
    follow_up_questions: Optional[List[str]]


class ToolCallInput(TypedDict):
    """Structure for tool call input."""
    args: Dict[str, str]
    type: Literal["tool_call"]
    id: str
    name: str


def create_tavily_search_tool(config: SearchConfig) -> TavilySearchResults:
    """
    Create a configured TavilySearchResults tool.

    Args:
        config: Configuration parameters for the search tool.

    Returns:
        A configured TavilySearchResults instance.
    """
    return TavilySearchResults(
        max_results=config["max_results"],
        search_depth=config["search_depth"],
        include_answer=config["include_answer"],
        include_raw_content=config["include_raw_content"],
        include_images=config["include_images"],
        include_domains=config.get("include_domains", []),
        exclude_domains=config.get("exclude_domains", []),
        redis_config={"host": config["redis_host"],
                      "port": config["redis_port"]}
    )


def create_tavily_answer_tool() -> TavilyAnswer:
    """
    Create a configured TavilyAnswer tool.

    Returns:
        A configured TavilyAnswer instance.
    """
    return TavilyAnswer()


async def perform_async_search(
    query: str,
    config: SearchConfig,
    run_manager: Optional[AsyncCallbackManagerForToolRun] = None
) -> Tuple[List[SearchResult], SearchArtifact]:
    """
    Perform an asynchronous search using TavilySearchResults.

    Args:
        query: The search query string.
        config: Configuration for the search tool.
        run_manager: Optional callback manager for async execution.

    Returns:
        A tuple of cleaned search results and the full artifact.
    """
    tool = create_tavily_search_tool(config)
    result, artifact = await tool._arun(query, run_manager)
    if isinstance(result, str):
        raise RuntimeError(f"Search failed: {result}")
    return result, artifact


def perform_sync_search(
    query: str,
    config: SearchConfig,
    run_manager: Optional[CallbackManagerForToolRun] = None
) -> Tuple[List[SearchResult], SearchArtifact]:
    """
    Perform a synchronous search using TavilySearchResults.

    Args:
        query: The search query string.
        config: Configuration for the search tool.
        run_manager: Optional callback manager for sync execution.

    Returns:
        A tuple of cleaned search results and the full artifact.
    """
    tool = create_tavily_search_tool(config)
    result, artifact = tool._run(query, run_manager)
    if isinstance(result, str):
        raise RuntimeError(f"Search failed: {result}")
    return result, artifact


async def get_answer_async(query: str) -> str:
    """
    Retrieve an answer asynchronously using TavilyAnswer.

    Args:
        query: The search query string.

    Returns:
        The answer string.
    """
    tool = create_tavily_answer_tool()
    result = await tool._arun(query)
    if isinstance(result, str) and result.startswith("Error"):
        raise RuntimeError(f"Answer retrieval failed: {result}")
    return result


def get_answer_sync(query: str) -> str:
    """
    Retrieve an answer synchronously using TavilyAnswer.

    Args:
        query: The search query string.

    Returns:
        The answer string.
    """
    tool = create_tavily_answer_tool()
    result = tool._run(query)
    if isinstance(result, str) and result.startswith("Error"):
        raise RuntimeError(f"Answer retrieval failed: {result}")
    return result


def handle_tool_call(tool_call: ToolCallInput) -> ToolMessage:
    """
    Handle a tool call for TavilySearchResults.

    Args:
        tool_call: The tool call input with query and metadata.

    Returns:
        A ToolMessage containing the search results and artifact.
    """
    if tool_call["name"] != "tavily_search_results_json":
        raise ValueError(f"Unsupported tool name: {tool_call['name']}")

    config: SearchConfig = {
        "max_results": 4,
        "search_depth": "advanced",
        "include_answer": True,
        "include_raw_content": False,
        "include_images": True,
        "redis_host": "localhost",
        "redis_port": 3103
    }
    tool = create_tavily_search_tool(config)
    result = tool.invoke(tool_call)
    return result


load_dotenv()

# Type definitions


class TavilyApiSearchResult(TypedDict):
    title: str
    url: str
    content: str
    score: float


class TavilyResponse(TypedDict):
    query: str
    results: list[TavilyApiSearchResult]


ResponseFormat = Literal["markdown", "json"]


def format_query_results(json_data: TavilyResponse, response_format: ResponseFormat = "markdown") -> Union[str, Dict]:
    """
    Formats Tavily search results into a specified format.

    Args:
        json_data: Dictionary containing Tavily query results
        response_format: Output format, either 'markdown' or 'json'

    Returns:
        Formatted results in specified format
    """
    if response_format == "json":
        return json_data  # ✅ return dict, not string

    formatted_text = []

    # Add the main query
    query = json_data.get("query", "No query provided.")
    formatted_text.append(f"### Query\n{query}\n\n---\n")

    # Add the results
    results = json_data.get("results", [])
    formatted_text.append("### Results\n")

    if not results:
        formatted_text.append("No results found.\n")
    else:
        for i, result in enumerate(results, 1):
            title = result.get("title", "No title")
            url = result.get("url", "No URL")
            content = result.get("content", "No content available.")
            score = result.get("score", "Not available")

            formatted_text.append(f"{i}. **Title**: {title}\n")
            formatted_text.append(f"   **URL**: {url}\n")
            formatted_text.append(f"   **Content**: {content}\n")
            formatted_text.append(f"   **Score**: {score}\n\n")

    return "".join(formatted_text)


def tavily_search(query: str, response_format: ResponseFormat = "markdown", max_results: int = 10) -> Union[str, TavilyResponse]:
    """
    Performs a web search using the Tavily API with raw httpx calls.

    Args:
        query: Search query string
        response_format: Output format, either 'markdown' or 'json'
        max_results: Maximum number of search results to return (default: 10)

    Returns:
        Formatted search results string in specified format
    """
    # Get API key from environment
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY environment variable is required")

    # Set up headers and endpoint
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    base_url = "https://api.tavily.com"
    endpoint = f"{base_url}/search"

    # Prepare request payload
    payload = {"query": query, "max_results": max_results}

    # Make the API request using httpx
    with httpx.Client() as client:
        try:
            response = client.post(
                endpoint, headers=headers, json=payload, timeout=30.0
            )

            # Check if the request was successful
            if response.status_code == 200:
                response_data: TavilyResponse = response.json()

                # Print raw JSON response
                print("\nTavily Raw Response:")
                print(response_data)

                # Format results
                formatted_text = format_query_results(
                    response_data, response_format)

                return formatted_text
            else:
                error_msg = f"API request failed with status {response.status_code}: {response.text}"
                print(f"Error: {error_msg}")
                return error_msg

        except httpx.RequestError as e:
            error_msg = f"Request error: {str(e)}"
            print(f"Error: {error_msg}")
            return error_msg
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP error: {e.response.status_code} - {e.response.text}"
            print(f"Error: {error_msg}")
            return error_msg
