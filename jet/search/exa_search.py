import os
import httpx
from typing import TypedDict, Literal, Any, Dict
from jet.logger import logger
from swarms.utils.any_to_str import any_to_str


class ExaSummary(TypedDict):
    answer: str


class ExaContents(TypedDict, total=False):
    text: bool
    summary: Dict[str, Any]
    context: Dict[str, Any]


class ExaPayload(TypedDict):
    query: str
    type: str
    numResults: int
    contents: ExaContents


ResponseFormat = Literal["json", "markdown"]


def exa_search(
    query: str,
    characters: int = 200,
    sources: int = 3,
    response_format: ResponseFormat = "json",
) -> str:
    """
    Exa Web Search Tool
    Args:
        query (str): Natural language search query
        characters (int): Max characters for context
        sources (int): Number of results
        response_format (Literal): "json" or "markdown"
    Returns:
        str: Search results in requested format
    """
    api_key = os.getenv("EXA_API_KEY")
    if not api_key:
        raise ValueError("EXA_API_KEY environment variable is not set")

    headers = {
        "x-api-key": api_key,
        "content-type": "application/json",
    }

    payload: ExaPayload = {
        "query": query,
        "type": "auto",
        "numResults": sources,
        "contents": {
            "text": True,
            "summary": {
                "schema": {
                    "type": "object",
                    "required": ["answer"],
                    "additionalProperties": False,
                    "properties": {
                        "answer": {
                            "type": "string",
                            "description": "Key insights and findings from the search result",
                        }
                    },
                }
            },
            "context": {"maxCharacters": characters},
        },
    }

    try:
        logger.info(f"[SEARCH] Executing Exa search for: {query[:50]}...")
        response = httpx.post(
            "https://api.exa.ai/search",
            json=payload,
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        json_data = response.json()

        if response_format == "markdown":
            results = json_data.get("results", [])
            if not results:
                return "No results found.\n"
            formatted = []
            for i, r in enumerate(results, 1):
                title = r.get("title", "No title")
                url = r.get("url", "No URL")
                content = r.get("text", "No content available.")
                formatted.append(f"{i}. **Title**: {title}\n")
                formatted.append(f"   **URL**: {url}\n")
                formatted.append(f"   **Content**: {content}\n\n")
            return "".join(formatted)

        return any_to_str(json_data)

    except Exception as e:
        logger.error(f"Exa search failed: {e}")
        return f"Search failed: {str(e)}. Please try again."
