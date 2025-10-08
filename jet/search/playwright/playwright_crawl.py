from typing import Any, Dict, List, Literal, Optional, Type
from langchain_core.callbacks import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, ConfigDict, Field
from playwright.async_api import async_playwright
from urllib.parse import urljoin, urlparse
import asyncio
import re
from bs4 import BeautifulSoup
import markdownify

from jet.scrapers.browser.config import PLAYWRIGHT_CHROMIUM_EXECUTABLE

class PlaywrightCrawlInput(BaseModel):
    """Input for PlaywrightCrawl"""
    url: str = Field(
        description="The root URL to begin the crawl."
    )
    max_depth: Optional[int] = Field(
        default=1,
        description="""Max depth of the crawl. Defines how many link hops from the root URL the crawler can explore.
        Use higher values (e.g., 2 or more) when:
        1. Crawling large websites to get a comprehensive overview of their structure.
        2. Exploring websites with many interconnected pages.
        Set to 1 when:
        1. Crawling a single page or staying local to the base URL.
        2. Needing quick results with minimal exploration.
        Must be greater than 0.
        Default is 1.
        """
    )
    max_breadth: Optional[int] = Field(
        default=20,
        description="""Max number of links to follow per page (per level of the crawl tree).
        Uses Breadth-First Search (BFS) where depth refers to link hops from the root URL.
        A page directly linked from the root is at BFS depth 1, regardless of URL structure.
        Increase this when:
        1. You want to explore many links from each page (e.g., for broad site coverage).
        Must be greater than 0.
        Default is 20.
        """
    )
    limit: Optional[int] = Field(
        default=50,
        description="""Total number of links the crawler will process before stopping.
        Use to control the scope of the crawl and prevent excessive resource usage.
        Must be greater than 0.
        Default is 50.
        """
    )
    instructions: Optional[str] = Field(
        default=None,
        description="""Natural language instructions to guide the crawler's focus.
        Use when the user specifies a specific goal or content type.
        Example: For 'Find all JavaScript SDK documentation from Tavily', set to 'JavaScript SDK documentation'.
        Helps the crawler prioritize relevant pages when combined with categories or path filters.
        Default is None (no specific instructions).
        """
    )
    select_paths: Optional[List[str]] = Field(
        default=None,
        description="""Regex patterns to select only URLs with specific path patterns.
        Use when the user explicitly requests a specific path from a website.
        Examples:
        - 'Crawl only the /api/v1 path' → ['/api/v1.*']
        - 'Crawl only the /documentation path' → ['/documentation/.*']
        Default is None (no path restriction).
        """
    )
    select_domains: Optional[List[str]] = Field(
        default=None,
        description="""Regex patterns to select only URLs from specific domains or subdomains.
        Use when the user explicitly requests a specific domain or subdomain.
        Example: 'Crawl only the docs.tavily.com subdomain' → ['^docs\\.tavily\\.com$']
        Default is None (no domain restriction).
        """
    )
    exclude_paths: Optional[List[str]] = Field(
        default=None,
        description="""Regex patterns to exclude URLs with specific path patterns.
        Use when the user explicitly requests to avoid specific paths.
        Examples:
        - 'Crawl example.com but exclude the /api/v1 path' → ['/api/v1.*']
        - 'Crawl example.com but exclude the /documentation path' → ['/documentation/.*']
        Default is None (no path exclusion).
        """
    )
    exclude_domains: Optional[List[str]] = Field(
        default=None,
        description="""Regex patterns to exclude URLs from specific domains or subdomains.
        Use when the user explicitly requests to avoid specific domains or subdomains.
        Example: 'Crawl tavily.com but exclude the docs.tavily.com subdomain' → ['^docs\\.tavily\\.com$']
        Default is None (no domain exclusion).
        """
    )
    allow_external: Optional[bool] = Field(
        default=False,
        description="""Determines whether the crawler can follow links to external domains.
        Set to True when the user explicitly allows external links (e.g., 'Crawl all related sites').
        Set to False to restrict crawling to the root URL's domain.
        Default is False (no external links).
        """
    )
    include_images: Optional[bool] = Field(
        default=True,
        description="""Determines whether to include image URLs in the crawl results.
        Set to True when images are relevant to the user's request (e.g., 'Crawl a site for product images').
        Default is True to leverage Playwright's ability to extract visual content.
        """
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
        description="""Directs the crawler to focus on specific website categories.
        Set to the category that best matches the user's request. Use the following guide:
        - Careers: Job listings, open positions, career information.
        - Blogs: Blog posts, news articles, editorial content.
        - Documentation: Technical docs, user guides, API references.
        - About: Company background, mission statements, team info.
        - Pricing: Product or service pricing, plans, cost comparisons.
        - Community: Forums, discussion boards, user groups.
        - Developers: Developer portals, SDKs, API documentation.
        - Contact: Contact info, support forms, customer service.
        - Media: Press releases, media kits, newsrooms, multimedia.
        Examples:
        - 'Crawl apple.com for career opportunities' → ['Careers']
        - 'Crawl tavily.com for API documentation' → ['Documentation']
        Default is None (no category restriction).
        """
    )
    extract_depth: Optional[Literal["basic", "advanced"]] = Field(
        default="basic",
        description="""Controls the thoroughness of web content extraction.
        Use 'basic' for faster extraction of main text content, suitable for simple pages.
        Use 'advanced' for comprehensive extraction, including tables, embedded elements, and complex structures.
        Always use 'advanced' for LinkedIn, YouTube, or other dynamic websites for optimal results.
        'advanced' may increase response time but improves content coverage.
        Default is 'basic'.
        """
    )
    include_favicon: Optional[bool] = Field(
        default=True,
        description="""Determines whether to include favicon URLs for each crawled page.
        When enabled, each result includes the website's favicon URL, useful for:
        - Building rich UI interfaces with visual website indicators
        - Providing visual cues about the source's credibility or brand
        - Creating bookmark-like displays with recognizable site icons
        Default is True to enhance result presentation.
        """
    )
    format: Optional[Literal["markdown", "text"]] = Field(
        default="markdown",
        description="""The format of the extracted web page content.
        'markdown' returns content in markdown format, suitable for structured rendering.
        'text' returns plain text, which may increase latency due to additional processing.
        Default is 'markdown'.
        """
    )

class PlaywrightCrawlAPIWrapper(BaseModel):
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
        include_images: Optional[bool],
        categories: Optional[List[Literal[
            "Careers", "Blogs", "Documentation", "About", "Pricing",
            "Community", "Developers", "Contact", "Media"
        ]]],
        extract_depth: Optional[Literal["basic", "advanced"]],
        include_favicon: Optional[bool],
        format: Optional[str],
    ) -> Dict[str, Any]:
        """Crawl a website using Playwright asynchronously."""
        visited = set()
        results = []
        base_domain = urlparse(url).netloc

        async def matches_criteria(url: str, content: str) -> bool:
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
            # Check categories
            if categories and not any(category.lower() in content.lower() for category in categories):
                return False
            return True

        async def extract_content(page, url: str) -> Dict[str, Any]:
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
            raw_content = soup.get_text(strip=True) if extract_depth == "basic" else str(soup)
            if format == "markdown":
                raw_content = markdownify.markdownify(raw_content)

            return {
                "url": url,
                "raw_content": raw_content,
                "images": images,
                "favicon": favicon
            }

        async def crawl(url: str, depth: int):
            if depth > (max_depth or 1) or len(results) >= (limit or 50) or url in visited:
                return
            visited.add(url)

            async with async_playwright() as p:
                browser = await p.chromium.launch(executable_path=PLAYWRIGHT_CHROMIUM_EXECUTABLE)
                page = await browser.new_page()
                try:
                    await page.goto(url, timeout=10000)
                    content = await page.content()
                    if await matches_criteria(url, content):
                        result = await extract_content(page, url)
                        results.append(result)
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
        include_images: Optional[bool],
        categories: Optional[List[Literal[
            "Careers", "Blogs", "Documentation", "About", "Pricing",
            "Community", "Developers", "Contact", "Media"
        ]]],
        extract_depth: Optional[Literal["basic", "advanced"]],
        include_favicon: Optional[bool],
        format: Optional[str],
    ) -> Dict[str, Any]:
        """Synchronous wrapper for async crawling."""
        return asyncio.run(self.raw_results_async(
            url, max_depth, max_breadth, limit, instructions,
            select_paths, select_domains, exclude_paths, exclude_domains,
            allow_external, include_images, categories, extract_depth,
            include_favicon, format
        ))

class PlaywrightCrawl(BaseTool):
    """Tool that crawls websites using Playwright with dynamically settable parameters."""
    name: str = "playwright_crawl"
    description: str = """A web crawler using Playwright to extract content from websites."""
    args_schema: Type[BaseModel] = PlaywrightCrawlInput
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
    include_images: bool = True
    categories: Optional[List[Literal[
        "Careers", "Blogs", "Documentation", "About", "Pricing",
        "Community", "Developers", "Contact", "Media"
    ]]] = None
    extract_depth: Optional[Literal["basic", "advanced"]] = None
    include_favicon: bool = True
    format: Optional[str] = None
    api_wrapper: PlaywrightCrawlAPIWrapper = Field(default_factory=PlaywrightCrawlAPIWrapper)

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
        include_images: bool = True,
        categories: Optional[List[Literal[
            "Careers", "Blogs", "Documentation", "About", "Pricing",
            "Community", "Developers", "Contact", "Media"
        ]]] = None,
        extract_depth: Optional[Literal["basic", "advanced"]] = None,
        include_favicon: bool = True,
        format: Optional[str] = None,
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
                include_images=self.include_images if self.include_images else include_images,
                categories=self.categories if self.categories else categories,
                extract_depth=self.extract_depth if self.extract_depth else extract_depth,
                include_favicon=self.include_favicon if self.include_favicon else include_favicon,
                format=self.format if self.format else format,
            )
            if not raw_results.get("results", []):
                search_params = {
                    "instructions": instructions,
                    "select_paths": select_paths,
                    "select_domains": select_domains,
                    "exclude_paths": exclude_paths,
                    "exclude_domains": exclude_domains,
                    "categories": categories,
                    "format": format,
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
        include_images: bool = True,
        categories: Optional[List[Literal[
            "Careers", "Blogs", "Documentation", "About", "Pricing",
            "Community", "Developers", "Contact", "Media"
        ]]] = None,
        extract_depth: Optional[Literal["basic", "advanced"]] = None,
        include_favicon: bool = True,
        format: Optional[str] = None,
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
                include_images=self.include_images if self.include_images else include_images,
                categories=self.categories if self.categories else categories,
                extract_depth=self.extract_depth if self.extract_depth else extract_depth,
                include_favicon=self.include_favicon if self.include_favicon else include_favicon,
                format=self.format if self.format else format,
            )
            if not raw_results.get("results", []):
                search_params = {
                    "instructions": instructions,
                    "select_paths": select_paths,
                    "select_domains": select_domains,
                    "exclude_paths": exclude_paths,
                    "exclude_domains": exclude_domains,
                    "categories": categories,
                    "format": format,
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
    format = params.get("format")
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
    if format:
        suggestions.append("Try a different format (markdown or text)")
    return suggestions
