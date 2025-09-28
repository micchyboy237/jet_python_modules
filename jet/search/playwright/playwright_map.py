from typing import Any, Dict, List, Literal, Optional, Type
from langchain_core.callbacks import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, ConfigDict, Field
from playwright.async_api import async_playwright
from urllib.parse import urljoin, urlparse
import asyncio
import re

class PlaywrightMapInput(BaseModel):
    """Input for PlaywrightMap"""
    url: str = Field(description="The root URL to begin the mapping.")
    max_depth: Optional[int] = Field(
        default=1,
        description="""Max depth of the mapping. Defines how far from the base URL the crawler can explore.
        Increase this parameter when:
        1. To map large websites and get a comprehensive overview of its structure.
        2. To map a website that has a lot of links to other pages.
        Set this parameter to 1 when:
        1. To stay local to the base_url
        2. To map a single page
        max_depth must be greater than 0
        """,
    )
    max_breadth: Optional[int] = Field(
        default=20,
        description="""Max number of links to follow per level of the tree (i.e., per page).
        Increase this parameter when:
        1. You want many links from each page to be mapped.
        max_breadth must be greater than 0
        """,
    )
    limit: Optional[int] = Field(
        default=50,
        description="""Total number of links the mapper will process before stopping.
        limit must be greater than 0
        """,
    )
    instructions: Optional[str] = Field(
        default=None,
        description="""Natural language instructions for the mapper.
        ex. "Javascript SDK documentation"
        """,
    )
    select_paths: Optional[List[str]] = Field(
        default=None,
        description="""Regex patterns to select only URLs with specific path patterns.
        ex. ["/api/v1.*"] 
        """,
    )
    select_domains: Optional[List[str]] = Field(
        default=None,
        description="""Regex patterns to select only URLs from specific domains or subdomains.
        ex. ["^docs\\.tavily\\.com$"]
        """,
    )
    exclude_paths: Optional[List[str]] = Field(
        default=None,
        description="""Regex patterns to exclude URLs from the map with specific path patterns.
        ex. ["/documentation/.*"]
        """,
    )
    exclude_domains: Optional[List[str]] = Field(
        default=None,
        description="""Regex patterns to exclude URLs from specific domains or subdomains.
        ex. ["^docs\\.tavily\\.com$"]
        """,
    )
    allow_external: Optional[bool] = Field(
        default=False,
        description="""Allow the crawler to follow external links.""",
    )
    categories: Optional[
        List[
            Literal[
                "Careers",
                "Blogs",
                "Documentation",
                "About",
                "Pricing",
                "Community",
                "Developers",
                "Contact",
                "Media",
            ]
        ]
    ] = Field(
        default=None,
        description="""Direct the crawler to crawl specific categories of a website.""",
    )

class PlaywrightMapAPIWrapper(BaseModel):
    """Wrapper for Playwright-based web crawler."""

    model_config = ConfigDict(
        extra="forbid",
    )

    async def raw_results_async(
        self,
        url: str,
        max_depth: Optional[int],
        max_breadth: Optional[int],
        limit: Optional[int],
        instructions: Optional[str],
        select_paths: Optional[List[str]],
        select_domains: Optional[List[str]],
        exclude_paths: Optional[List[str]],
        exclude_domains: Optional[List[str]],
        allow_external: Optional[bool],
        categories: Optional[List[Literal[
            "Careers", "Blogs", "Documentation", "About", "Pricing",
            "Community", "Developers", "Contact", "Media"
        ]]],
    ) -> Dict[str, Any]:
        """Crawl a website using Playwright asynchronously."""
        visited = set()
        results = []
        base_domain = urlparse(url).netloc

        async def matches_criteria(url: str) -> bool:
            path = urlparse(url).path
            domain = urlparse(url).netloc

            # Check domain restrictions
            if not allow_external and domain != base_domain:
                return False
            if select_domains and not any(re.match(pattern, domain) for pattern in select_domains):
                return False
            if exclude_domains and any(re.match(pattern, domain) for pattern in exclude_domains):
                return False
            # Check path restrictions
            if select_paths and not any(re.match(pattern, path) for pattern in select_paths):
                return False
            if exclude_paths and any(re.match(pattern, path) for pattern in exclude_paths):
                return False
            # Check categories (basic keyword matching)
            if categories:
                async with async_playwright() as p:
                    browser = await p.chromium.launch()
                    page = await browser.new_page()
                    try:
                        await page.goto(url, timeout=10000)
                        content = await page.content()
                        for category in categories:
                            if category.lower() in content.lower():
                                return True
                        return False
                    finally:
                        await browser.close()
            return True

        async def crawl(url: str, depth: int):
            if depth > (max_depth or 1) or len(results) >= (limit or 50) or url in visited:
                return
            visited.add(url)

            async with async_playwright() as p:
                browser = await p.chromium.launch()
                page = await browser.new_page()
                try:
                    await page.goto(url, timeout=10000)
                    if await matches_criteria(url):
                        results.append({"url": url})
                    links = await page.query_selector_all('a')
                    hrefs = [
                        urljoin(url, await link.get_attribute('href') or '')
                        for link in links[:max_breadth or 20]
                    ]
                    for href in hrefs:
                        if href not in visited and len(results) < (limit or 50):
                            await crawl(href, depth + 1)
                except Exception:
                    pass
                finally:
                    await browser.close()

        start_time = asyncio.get_event_loop().time()
        await crawl(url, 1)
        response_time = asyncio.get_event_loop().time() - start_time

        return {
            "base_url": url,
            "results": results,
            "response_time": response_time
        }

    def raw_results(
        self,
        url: str,
        max_depth: Optional[int],
        max_breadth: Optional[int],
        limit: Optional[int],
        instructions: Optional[str],
        select_paths: Optional[List[str]],
        select_domains: Optional[List[str]],
        exclude_paths: Optional[List[str]],
        exclude_domains: Optional[List[str]],
        allow_external: Optional[bool],
        categories: Optional[List[Literal[
            "Careers", "Blogs", "Documentation", "About", "Pricing",
            "Community", "Developers", "Contact", "Media"
        ]]],
    ) -> Dict[str, Any]:
        """Synchronous wrapper for async crawling."""
        return asyncio.run(self.raw_results_async(
            url, max_depth, max_breadth, limit, instructions,
            select_paths, select_domains, exclude_paths, exclude_domains,
            allow_external, categories
        ))

class PlaywrightMap(BaseTool):
    """Tool that crawls websites using Playwright with dynamically settable parameters."""
    name: str = "playwright_map"
    description: str = """A web mapping tool using Playwright to create a structured map of website URLs."""
    args_schema: Type[BaseModel] = PlaywrightMapInput
    handle_tool_error: bool = True
    max_depth: Optional[int] = None
    max_breadth: Optional[int] = None
    limit: Optional[int] = None
    instructions: Optional[str] = None
    select_paths: Optional[List[str]] = None
    select_domains: Optional[List[str]] = None
    exclude_paths: Optional[List[str]] = None
    exclude_domains: Optional[List[str]] = None
    allow_external: Optional[bool] = None
    categories: Optional[List[Literal[
        "Careers", "Blogs", "Documentation", "About", "Pricing",
        "Community", "Developers", "Contact", "Media"
    ]]] = None
    api_wrapper: PlaywrightMapAPIWrapper = Field(default_factory=PlaywrightMapAPIWrapper)

    def _run(
        self,
        url: str,
        max_depth: Optional[int] = None,
        max_breadth: Optional[int] = None,
        limit: Optional[int] = None,
        instructions: Optional[str] = None,
        select_paths: Optional[List[str]] = None,
        select_domains: Optional[List[str]] = None,
        exclude_paths: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        allow_external: Optional[bool] = None,
        categories: Optional[List[Literal[
            "Careers", "Blogs", "Documentation", "About", "Pricing",
            "Community", "Developers", "Contact", "Media"
        ]]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        try:
            raw_results = self.api_wrapper.raw_results(
                url=url,
                max_depth=self.max_depth if self.max_depth else max_depth,
                max_breadth=self.max_breadth if self.max_breadth else max_breadth,
                limit=self.limit if self.limit else limit,
                instructions=self.instructions if self.instructions else instructions,
                select_paths=self.select_paths if self.select_paths else select_paths,
                select_domains=self.select_domains if self.select_domains else select_domains,
                exclude_paths=self.exclude_paths if self.exclude_paths else exclude_paths,
                exclude_domains=self.exclude_domains if self.exclude_domains else exclude_domains,
                allow_external=self.allow_external if self.allow_external else allow_external,
                categories=self.categories if self.categories else categories,
            )
            if not raw_results.get("results", []):
                search_params = {
                    "instructions": instructions,
                    "select_paths": select_paths,
                    "select_domains": select_domains,
                    "exclude_paths": exclude_paths,
                    "exclude_domains": exclude_domains,
                    "categories": categories,
                }
                suggestions = _generate_suggestions(search_params)
                error_message = (
                    f"No crawl results found for '{url}'. "
                    f"Suggestions: {', '.join(suggestions)}. "
                    f"Try modifying your crawl parameters with one of these approaches."
                )
                raise ToolException(error_message)
            return raw_results
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        url: str,
        max_depth: Optional[int] = None,
        max_breadth: Optional[int] = None,
        limit: Optional[int] = None,
        instructions: Optional[str] = None,
        select_paths: Optional[List[str]] = None,
        select_domains: Optional[List[str]] = None,
        exclude_paths: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        allow_external: Optional[bool] = None,
        categories: Optional[List[Literal[
            "Careers", "Blogs", "Documentation", "About", "Pricing",
            "Community", "Developers", "Contact", "Media"
        ]]] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        try:
            raw_results = await self.api_wrapper.raw_results_async(
                url=url,
                max_depth=self.max_depth if self.max_depth else max_depth,
                max_breadth=self.max_breadth if self.max_breadth else max_breadth,
                limit=self.limit if self.limit else limit,
                instructions=self.instructions if self.instructions else instructions,
                select_paths=self.select_paths if self.select_paths else select_paths,
                select_domains=self.select_domains if self.select_domains else select_domains,
                exclude_paths=self.exclude_paths if self.exclude_paths else exclude_paths,
                exclude_domains=self.exclude_domains if self.exclude_domains else exclude_domains,
                allow_external=self.allow_external if self.allow_external else allow_external,
                categories=self.categories if self.categories else categories,
            )
            if not raw_results.get("results", []):
                search_params = {
                    "instructions": instructions,
                    "select_paths": select_paths,
                    "select_domains": select_domains,
                    "exclude_paths": exclude_paths,
                    "exclude_domains": exclude_domains,
                    "categories": categories,
                }
                suggestions = _generate_suggestions(search_params)
                error_message = (
                    f"No crawl results found for '{url}'. "
                    f"Suggestions: {', '.join(suggestions)}. "
                    f"Try modifying your crawl parameters with one of these approaches."
                )
                raise ToolException(error_message)
            return raw_results
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

def _generate_suggestions(params: Dict[str, Any]) -> List[str]:
    """Generate helpful suggestions based on the failed crawl parameters."""
    suggestions = []
    instructions = params.get("instructions")
    select_paths = params.get("select_paths")
    select_domains = params.get("select_domains")
    exclude_paths = params.get("exclude_paths")
    exclude_domains = params.get("exclude_domains")
    categories = params.get("categories")
    if instructions:
        suggestions.append("Try more concise instructions")
    if select_paths:
        suggestions.append("Remove select_paths argument")
    if select_domains:
        suggestions.append("Remove select_domains argument")
    if exclude_paths:
        suggestions.append("Remove exclude_paths argument")
    if exclude_domains:
        suggestions.append("Remove exclude_domains argument")
    if categories:
        suggestions.append("Remove categories argument")
    return suggestions
