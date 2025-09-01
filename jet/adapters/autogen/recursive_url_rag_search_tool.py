import asyncio
from typing import Dict, List, Mapping, Optional
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from autogen_core.tools import ToolResult, TextResultContent
from autogen_core import CancellationToken
from pydantic import BaseModel, Field
import logging

# Setup logging
logger = logging.getLogger(__name__)


class RecursiveUrlRagSearchInput(BaseModel):
    query: str = Field(...,
                       description="The search query to filter relevant content")
    start_url: str = Field(...,
                           description="The starting URL for recursive crawling")
    max_depth: int = Field(
        2, description="Maximum recursion depth for URL crawling")
    max_urls: int = Field(10, description="Maximum number of URLs to process")


class RecursiveUrlRagSearchTool:
    """A tool for recursively loading URLs and performing RAG search based on a query."""

    def __init__(self, workbench: "McpWorkbench"):
        self.workbench = workbench
        self.schema = {
            "name": "recursive_url_rag_search",
            "description": "Recursively loads URLs starting from a given URL and performs a RAG search to find content relevant to a query.",
            "parameters": RecursiveUrlRagSearchInput.schema()
        }

    async def run(
        self,
        arguments: Mapping[str, any],
        cancellation_token: Optional[CancellationToken] = None
    ) -> ToolResult:
        """Execute the recursive URL loading and RAG search."""
        args = RecursiveUrlRagSearchInput(**arguments)
        visited_urls = set()
        relevant_content = []

        async def crawl_url(url: str, depth: int) -> None:
            logger.debug(f"Crawling URL: {url} at depth {depth}")
            if depth > args.max_depth or len(visited_urls) >= args.max_urls or url in visited_urls:
                logger.debug(
                    f"Skipping URL: {url} (depth: {depth}, visited: {len(visited_urls)})")
                return
            visited_urls.add(url)

            # Call the MCP server's fetch tool
            fetch_result = await self.workbench.call_tool(
                name="fetch",
                arguments={"url": url},
                cancellation_token=cancellation_token
            )
            logger.debug(
                f"Fetch result for {url}: {fetch_result.result}, is_error: {fetch_result.is_error}")

            # Extract text content from fetch result, stripping HTML tags
            raw_content = ""
            for part in fetch_result.result:
                if isinstance(part, TextResultContent):
                    raw_content += part.content
                if fetch_result.is_error:
                    logger.debug(f"Fetch error for {url}")
                    return

            # Parse raw content to extract clean text
            soup = BeautifulSoup(raw_content, "html.parser")
            content = soup.get_text(separator=" ", strip=True)
            logger.debug(f"Cleaned content for {url}: {content[:200]}")

            # Score content relevance based on query
            score = sum(content.lower().count(term.lower())
                        for term in args.query.split())
            logger.debug(f"Relevance score for {url}: {score}")
            if score > 0:
                relevant_content.append(
                    {"url": url, "content": content[:1000], "score": score})

            # Extract links for recursive crawling from raw content
            for link in soup.find_all("a", href=True):
                next_url = urljoin(url, link["href"])
                parsed_url = urlparse(next_url)
                if parsed_url.scheme in ("http", "https") and parsed_url.netloc:
                    logger.debug(f"Found link: {next_url}")
                    await crawl_url(next_url, depth + 1)

        await crawl_url(args.start_url, 0)
        logger.debug(
            f"Collected relevant content: {[item['url'] for item in relevant_content]}")

        # Sort by relevance score and format results
        relevant_content.sort(key=lambda x: x["score"], reverse=True)
        result_text = "\n\n".join(
            f"URL: {item['url']}\nRelevance Score: {item['score']}\nContent: {item['content'].strip()}"
            for item in relevant_content[:5]  # Limit to top 5 results
        ) or "No relevant content found."
        logger.debug(f"Formatted result_text: {result_text[:200]}")

        return ToolResult(
            name="recursive_url_rag_search",
            result=[TextResultContent(content=result_text)],
            is_error=False
        )
