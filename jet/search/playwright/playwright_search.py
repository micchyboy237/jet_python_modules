from typing import Any, Dict, List, Literal, Optional, Type, Union
from langchain_core.callbacks import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, ConfigDict, Field
from playwright.async_api import async_playwright
from urllib.parse import urljoin
import asyncio
from bs4 import BeautifulSoup
import markdownify
import re

class PlaywrightSearchInput(BaseModel):
    """Input for PlaywrightSearch"""
    query: str = Field(description="Search query to look up")
    include_domains: Optional[List[str]] = Field(
        default=None,
        description="""A list of domains to restrict search results to.""",
    )
    exclude_domains: Optional[List[str]] = Field(
        default=None,
        description="""A list of domains to exclude from search results.""",
    )
    search_depth: Optional[Literal["basic", "advanced"]] = Field(
        default="basic",
        description="""Controls search thoroughness and result comprehensiveness.""",
    )
    include_images: Optional[bool] = Field(
        default=False,
        description="""Determines if the search returns relevant images along with text results.""",
    )
    time_range: Optional[Literal["day", "week", "month", "year"]] = Field(
        default=None,
        description="""Limits results to content published within a specific timeframe.""",
    )
    topic: Optional[Literal["general", "news", "finance"]] = Field(
        default="general",
        description="""Specifies search category for optimized results.""",
    )
    include_favicon: Optional[bool] = Field(
        default=False,
        description="""Determines whether to include favicon URLs for each search result.""",
    )
    start_date: Optional[str] = Field(
        default=None,
        description="""Filters search results to include only content published on or after this date (YYYY-MM-DD).""",
    )
    end_date: Optional[str] = Field(
        default=None,
        description="""Filters search results to include only content published on or before this date (YYYY-MM-DD).""",
    )

class PlaywrightSearchAPIWrapper(BaseModel):
    """Wrapper for Playwright-based search engine."""

    model_config = ConfigDict(
        extra="forbid",
    )

    async def raw_results_async(
        self,
        query: str,
        max_results: Optional[int] = 5,
        search_depth: Optional[Literal["basic", "advanced"]] = "basic",
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        include_answer: Optional[Union[bool, Literal["basic", "advanced"]]] = False,
        include_raw_content: Optional[Union[bool, Literal["markdown", "text"]]] = "markdown",
        include_images: Optional[bool] = False,
        include_image_descriptions: Optional[bool] = False,
        include_favicon: Optional[bool] = False,
        topic: Optional[Literal["general", "news", "finance"]] = "general",
        time_range: Optional[Literal["day", "week", "month", "year"]] = None,
        country: Optional[str] = None,
        auto_parameters: Optional[bool] = False,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Perform a search using Playwright asynchronously."""
        results = []
        search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"

        async def matches_criteria(url: str, content: str) -> bool:
            domain = urlparse(url).netloc
            if include_domains and not any(re.match(pattern, domain) for pattern in include_domains):
                return False
            if exclude_domains and any(re.match(pattern, domain) for pattern in exclude_domains):
                return False
            if topic != "general" and topic.lower() not in content.lower():
                return False
            return True

        async def extract_content(url: str) -> Dict[str, Any]:
            async with async_playwright() as p:
                browser = await p.chromium.launch()
                page = await browser.new_page()
                try:
                    await page.goto(url, timeout=10000)
                    content = await page.content()
                    soup = BeautifulSoup(content, 'html.parser')
                    
                    # Remove scripts and styles
                    for element in soup(['script', 'style']):
                        element.decompose()
                    
                    # Extract favicon
                    favicon = None
                    if include_favicon:
                        favicon_link = soup.find('link', rel=['icon', 'shortcut icon'])
                        if favicon_link and favicon_link.get('href'):
                            favicon = urljoin(url, favicon_link['href'])

                    # Extract images
                    images = []
                    if include_images:
                        img_tags = soup.find_all('img')
                        images = [urljoin(url, img.get('src')) for img in img_tags if img.get('src')]

                    # Extract content based on depth
                    raw_content = None
                    if include_raw_content:
                        raw_content = soup.get_text(strip=True) if search_depth == "basic" else str(soup)
                        if include_raw_content == "markdown":
                            raw_content = markdownify.markdownify(raw_content)

                    # Extract title
                    title = soup.title.string if soup.title else url

                    # Extract snippet
                    snippet = soup.get_text(strip=True)[:200]

                    return {
                        "title": title,
                        "url": url,
                        "content": snippet,
                        "score": 0.9,  # Mock relevance score
                        "raw_content": raw_content,
                        "images": images,
                        "favicon": favicon
                    }
                except Exception as e:
                    return {"url": url, "error": str(e)}
                finally:
                    await browser.close()

        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            start_time = asyncio.get_event_loop().time()
            try:
                await page.goto(search_url, timeout=10000)
                links = await page.query_selector_all('a')
                hrefs = [
                    urljoin(search_url, await link.get_attribute('href') or '')
                    for link in links if await link.get_attribute('href')
                ]
                # Filter out Google-specific URLs
                hrefs = [href for href in hrefs if not href.startswith(('https://www.google.', 'https://accounts.google.'))]
                
                tasks = []
                for href in hrefs[:max_results or 5]:
                    async with browser.new_page() as new_page:
                        try:
                            await new_page.goto(href, timeout=10000)
                            content = await new_page.content()
                            if await matches_criteria(href, content):
                                tasks.append(extract_content(href))
                        except:
                            continue
                
                extracted_results = await asyncio.gather(*tasks, return_exceptions=True)
                for result in extracted_results:
                    if isinstance(result, dict) and "error" not in result:
                        results.append(result)
                
            finally:
                await browser.close()

        response_time = asyncio.get_event_loop().time() - start_time
        return {
            "query": query,
            "results": results[:max_results or 5],
            "images": [img for result in results for img in result.get("images", [])] if include_images else [],
            "response_time": response_time
        }

    def raw_results(
        self,
        query: str,
        max_results: Optional[int] = 5,
        search_depth: Optional[Literal["basic", "advanced"]] = "basic",
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        include_answer: Optional[Union[bool, Literal["basic", "advanced"]]] = False,
        include_raw_content: Optional[Union[bool, Literal["markdown", "text"]]] = "markdown",
        include_images: Optional[bool] = False,
        include_image_descriptions: Optional[bool] = False,
        include_favicon: Optional[bool] = False,
        topic: Optional[Literal["general", "news", "finance"]] = "general",
        time_range: Optional[Literal["day", "week", "month", "year"]] = None,
        country: Optional[str] = None,
        auto_parameters: Optional[bool] = False,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Synchronous wrapper for async search."""
        return asyncio.run(self.raw_results_async(
            query, max_results, search_depth, include_domains, exclude_domains,
            include_answer, include_raw_content, include_images, include_image_descriptions,
            include_favicon, topic, time_range, country, auto_parameters, start_date, end_date
        ))

class PlaywrightSearch(BaseTool):
    """Tool that performs web searches using Playwright."""
    name: str = "playwright_search"
    description: str = """A search engine using Playwright to retrieve web content based on a query."""
    args_schema: Type[BaseModel] = PlaywrightSearchInput
    handle_tool_error: bool = True
    auto_parameters: Optional[bool] = None
    include_domains: Optional[List[str]] = None
    exclude_domains: Optional[List[str]] = None
    search_depth: Optional[Literal["basic", "advanced"]] = None
    include_images: Optional[bool] = None
    time_range: Optional[Literal["day", "week", "month", "year"]] = None
    max_results: Optional[int] = None
    topic: Optional[Literal["general", "news", "finance"]] = None
    include_answer: Optional[Union[bool, Literal["basic", "advanced"]]] = None
    include_raw_content: Optional[Union[bool, Literal["markdown", "text"]]] = None
    include_image_descriptions: Optional[bool] = None
    country: Optional[str] = None
    include_favicon: Optional[bool] = None
    api_wrapper: PlaywrightSearchAPIWrapper = Field(default_factory=PlaywrightSearchAPIWrapper)

    def _run(
        self,
        query: str,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        search_depth: Optional[Literal["basic", "advanced"]] = None,
        include_images: Optional[bool] = None,
        time_range: Optional[Literal["day", "week", "month", "year"]] = None,
        topic: Optional[Literal["general", "news", "finance"]] = None,
        include_favicon: Optional[bool] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_results: Optional[int] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        try:
            raw_results = self.api_wrapper.raw_results(
                query=query,
                include_domains=self.include_domains if self.include_domains else include_domains,
                exclude_domains=self.exclude_domains if self.exclude_domains else exclude_domains,
                search_depth=self.search_depth if self.search_depth else search_depth,
                include_images=self.include_images if self.include_images else include_images,
                time_range=self.time_range if self.time_range else time_range,
                topic=self.topic if self.topic else topic,
                include_favicon=self.include_favicon if self.include_favicon else include_favicon,
                country=self.country,
                max_results=self.max_results if self.max_results is not None else max_results,
                include_answer=self.include_answer,
                include_raw_content=self.include_raw_content,
                include_image_descriptions=self.include_image_descriptions,
                auto_parameters=self.auto_parameters,
                start_date=start_date,
                end_date=end_date,
            )
            if not raw_results.get("results", []):
                search_params = {
                    "time_range": time_range,
                    "include_domains": include_domains,
                    "search_depth": search_depth,
                    "exclude_domains": exclude_domains,
                    "topic": topic,
                }
                suggestions = _generate_suggestions(search_params)
                error_message = (
                    f"No search results found for '{query}'. "
                    f"Suggestions: {', '.join(suggestions)}. "
                    f"Try modifying your search parameters with one of these approaches."
                )
                raise ToolException(error_message)
            return raw_results
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        query: str,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        search_depth: Optional[Literal["basic", "advanced"]] = None,
        include_images: Optional[bool] = None,
        time_range: Optional[Literal["day", "week", "month", "year"]] = None,
        topic: Optional[Literal["general", "news", "finance"]] = None,
        include_favicon: Optional[bool] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_results: Optional[int] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        try:
            raw_results = await self.api_wrapper.raw_results_async(
                query=query,
                include_domains=self.include_domains if self.include_domains else include_domains,
                exclude_domains=self.exclude_domains if self.exclude_domains else exclude_domains,
                search_depth=self.search_depth if self.search_depth else search_depth,
                include_images=self.include_images if self.include_images else include_images,
                time_range=self.time_range if self.time_range else time_range,
                topic=self.topic if self.topic else topic,
                include_favicon=self.include_favicon if self.include_favicon else include_favicon,
                country=self.country,
                max_results=self.max_results if self.max_results is not None else max_results,
                include_answer=self.include_answer,
                include_raw_content=self.include_raw_content,
                include_image_descriptions=self.include_image_descriptions,
                auto_parameters=self.auto_parameters,
                start_date=start_date,
                end_date=end_date,
            )
            if not raw_results.get("results", []):
                search_params = {
                    "time_range": time_range,
                    "include_domains": include_domains,
                    "search_depth": search_depth,
                    "exclude_domains": exclude_domains,
                    "topic": topic,
                }
                suggestions = _generate_suggestions(search_params)
                error_message = (
                    f"No search results found for '{query}'. "
                    f"Suggestions: {', '.join(suggestions)}. "
                    f"Try modifying your search parameters with one of these approaches."
                )
                raise ToolException(error_message)
            return raw_results
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

def _generate_suggestions(params: Dict[str, Any]) -> List[str]:
    """Generate helpful suggestions based on the failed search parameters."""
    suggestions = []
    search_depth = params.get("search_depth")
    exclude_domains = params.get("exclude_domains")
    include_domains = params.get("include_domains")
    time_range = params.get("time_range")
    topic = params.get("topic")
    if time_range:
        suggestions.append("Remove time_range argument")
    if include_domains:
        suggestions.append("Remove include_domains argument")
    if exclude_domains:
        suggestions.append("Remove exclude_domains argument")
    if search_depth == "basic":
        suggestions.append("Try a more detailed search using 'advanced' search_depth")
    if topic != "general":
        suggestions.append("Try a general search using 'general' topic")
    return suggestions
