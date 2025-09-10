from langchain_community.tools.tavily_search import TavilySearchResults
from .config import TAVILY_API_KEY


def get_web_search_tool():
    """Create web search tool."""
    if not TAVILY_API_KEY:
        raise ValueError("TAVILY_API_KEY not set in .env")
    return TavilySearchResults(max_results=3, api_key=TAVILY_API_KEY)
