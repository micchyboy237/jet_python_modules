import os
from typing import Any, Dict, TypedDict, Literal
import httpx
import json
from dotenv import load_dotenv
from rich.console import Console

console = Console()
load_dotenv()

# Type definitions


class SearchResult(TypedDict):
    title: str
    link: str
    snippet: str


class SerpAPIResponse(TypedDict):
    search_parameters: Dict[str, Any]
    organic_results: list[SearchResult]


ResponseFormat = Literal["markdown", "json"]


def format_serpapi_results(json_data: SerpAPIResponse, response_format: ResponseFormat = "markdown") -> str:
    """
    Formats SerpAPI search results into a specified format.

    Args:
        json_data: Dictionary containing SerpAPI search results
        response_format: Output format, either 'markdown' or 'json'

    Returns:
        Formatted string with search results in specified format
    """
    if response_format == "json":
        return json.dumps(json_data, indent=2)

    formatted_text = []

    # Extract query information
    search_params = json_data.get("search_parameters", {})
    query = search_params.get("q", "No query provided.")
    formatted_text.append(f"### Query\n{query}\n\n---\n")

    # Process organic results
    results = json_data.get("organic_results", [])
    formatted_text.append("### Results\n")

    if not results:
        formatted_text.append("No results found.\n")
    else:
        for i, result in enumerate(results, 1):
            title = result.get("title", "No title")
            url = result.get("link", "No URL")
            content = result.get("snippet", "No content available.")

            formatted_text.append(f"{i}. **Title**: {title}\n")
            formatted_text.append(f"   **URL**: {url}\n")
            formatted_text.append(f"   **Content**: {content}\n\n")

    return "".join(formatted_text)


def serpapi_search(query: str, response_format: ResponseFormat = "markdown") -> str:
    """
    Performs a web search using the SerpAPI Google Search API via HTTPX.

    Args:
        query: Search query string
        response_format: Output format, either 'markdown' or 'json'

    Returns:
        Formatted search results string in specified format
    """
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        raise ValueError("SERPAPI_API_KEY not found in environment variables.")

    params = {
        "api_key": api_key,
        "engine": "google",
        "q": query,
        "location": "Austin, Texas, United States",
        "google_domain": "google.com",
        "gl": "us",
        "hl": "en",
    }

    url = "https://serpapi.com/search"

    with httpx.Client(timeout=30.0) as client:
        try:
            response = client.get(url, params=params)
            response.raise_for_status()
            results: SerpAPIResponse = response.json()

            # Print raw JSON response
            console.print("\n[bold]SerpAPI Raw Response:[/bold]")
            console.print(results)

            # Format results
            formatted_text = format_serpapi_results(results, response_format)

            return formatted_text

        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP error: {e.response.status_code} - {e.response.text}"
            console.print(f"[red]Error: {error_msg}[/red]")
            return error_msg
        except httpx.RequestError as e:
            error_msg = f"Request error: {str(e)}"
            console.print(f"[red]Error: {error_msg}[/red]")
            return error_msg


# Example usage
if __name__ == "__main__":
    results = serpapi_search("Deepseek news")
    console.print("\n[bold]Formatted SerpAPI Results:[/bold]")
    console.print(results)
