from typing import Any, Dict, List, Literal, Optional, Type
from langchain_core.callbacks import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, ConfigDict, Field
from playwright.async_api import async_playwright
from urllib.parse import urljoin
import asyncio
from bs4 import BeautifulSoup
import markdownify

class PlaywrightExtractInput(BaseModel):
    """Input for PlaywrightExtract"""
    urls: List[str] = Field(description="List of URLs to extract")
    extract_depth: Optional[Literal["basic", "advanced"]] = Field(
        default="basic",
        description="""Controls the thoroughness of web content extraction.
        Use "basic" for faster extraction of main text content.
        Use "advanced" for comprehensive content including tables and embedded elements.""",
    )
    include_images: Optional[bool] = Field(
        default=True,
        description="""Determines whether to extract and include images from the source URLs.""",
    )
    include_favicon: Optional[bool] = Field(
        default=True,
        description="""Whether to include the favicon URL for each result.""",
    )
    format: Optional[str] = Field(
        default="markdown",
        description="""The format of the extracted web page content (markdown or text).""",
    )

class PlaywrightExtractAPIWrapper(BaseModel):
    """Wrapper for Playwright-based web content extractor."""

    model_config = ConfigDict(
        extra="forbid",
    )

    async def raw_results_async(
        self,
        urls: List[str],
        include_images: Optional[bool],
        include_favicon: Optional[bool],
        extract_depth: Optional[Literal["basic", "advanced"]],
        format: Optional[str],
    ) -> Dict[str, Any]:
        """Extract content from URLs using Playwright asynchronously."""
        results = []
        failed_results = []

        async def extract_content(url: str) -> Dict[str, Any]:
            async with async_playwright() as p:
                browser = await p.chromium.launch()
                page = await browser.new_page()
                try:
                    await page.goto(url, timeout=30000)  # Increased timeout to 30 seconds
                    content = await page.content()
                    if not content:
                        return {"url": url, "error": "Empty page content"}
                    soup = BeautifulSoup(content, 'html.parser')
                    for element in soup(['script', 'style']):
                        element.decompose()
                    favicon = None
                    if include_favicon:
                        favicon_link = soup.find('link', rel=['icon', 'shortcut icon'])
                        if favicon_link and favicon_link.get('href'):
                            favicon = urljoin(url, favicon_link['href'])
                    images = []
                    if include_images:
                        img_tags = soup.find_all('img')
                        images = [urljoin(url, img.get('src')) for img in img_tags if img.get('src')]
                    raw_content = soup.get_text(strip=True) if extract_depth == "basic" else str(soup)
                    if not raw_content:
                        return {"url": url, "error": "No content extracted after parsing"}
                    if format == "markdown":
                        raw_content = markdownify.markdownify(raw_content)
                    return {
                        "url": url,
                        "raw_content": raw_content,
                        "images": images,
                        "favicon": favicon
                    }
                except Exception as e:
                    return {"url": url, "error": f"Failed to extract content: {str(e)}"}
                finally:
                    await browser.close()

        start_time = asyncio.get_event_loop().time()
        tasks = [extract_content(url) for url in urls]
        extracted_results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in extracted_results:
            if "error" in result:
                failed_results.append(result)
            else:
                results.append(result)

        response_time = asyncio.get_event_loop().time() - start_time
        return {
            "results": results,
            "failed_results": failed_results,
            "response_time": response_time
        }

    def raw_results(
        self,
        urls: List[str],
        include_images: Optional[bool],
        include_favicon: Optional[bool],
        extract_depth: Optional[Literal["basic", "advanced"]],
        format: Optional[str],
    ) -> Dict[str, Any]:
        """Synchronous wrapper for async extraction."""
        return asyncio.run(self.raw_results_async(
            urls, include_images, include_favicon, extract_depth, format
        ))

class PlaywrightExtract(BaseTool):
    """Tool that extracts content from websites using Playwright."""
    name: str = "playwright_extract"
    description: str = """Extracts content from web pages using Playwright, supporting basic or advanced extraction."""
    args_schema: Type[BaseModel] = PlaywrightExtractInput
    handle_tool_error: bool = True
    extract_depth: Optional[Literal["basic", "advanced"]] = None
    include_images: bool = True
    include_favicon: bool = True
    format: Optional[str] = None
    apiwrapper: PlaywrightExtractAPIWrapper = Field(default_factory=PlaywrightExtractAPIWrapper)

    def _run(
        self,
        urls: List[str],
        extract_depth: Optional[Literal["basic", "advanced"]] = None,
        include_images: bool = True,
        include_favicon: bool = True,
        format: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        try:
            raw_results = self.apiwrapper.raw_results(
                urls=urls,
                extract_depth=self.extract_depth if self.extract_depth else extract_depth,
                include_images=self.include_images if self.include_images else include_images,
                include_favicon=self.include_favicon if self.include_favicon else include_favicon,
                format=self.format if self.format else format,
            )
            results = raw_results.get("results", [])
            failed_results = raw_results.get("failed_results", [])
            if not results or len(failed_results) == len(urls):
                search_params = {
                    "extract_depth": extract_depth,
                    "include_images": include_images,
                }
                suggestions = _generate_suggestions(search_params)
                error_message = (
                    f"No extracted results found for '{urls}'. "
                    f"Suggestions: {', '.join(suggestions)}. "
                    f"Try modifying your extract parameters with one of these approaches."
                )
                raise ToolException(error_message)
            return raw_results
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        urls: List[str],
        extract_depth: Optional[Literal["basic", "advanced"]] = None,
        include_images: bool = True,
        include_favicon: bool = True,
        format: Optional[str] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        try:
            raw_results = await self.apiwrapper.raw_results_async(
                urls=urls,
                extract_depth=self.extract_depth if self.extract_depth else extract_depth,
                include_images=self.include_images if self.include_images else include_images,
                include_favicon=self.include_favicon if self.include_favicon else include_favicon,
                format=self.format if self.format else format,
            )
            results = raw_results.get("results", [])
            failed_results = raw_results.get("failed_results", [])
            if not results and len(failed_results) == len(urls):
                search_params = {
                    "extract_depth": extract_depth,
                    "include_images": include_images,
                }
                suggestions = _generate_suggestions(search_params)
                error_message = (
                    f"No extracted results found for '{urls}'. "
                    f"Failed results: {failed_results}. "
                    f"Suggestions: {', '.join(suggestions)}. "
                    f"Try modifying your extract parameters or checking network connectivity."
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
    if params.get("extract_depth") and params["extract_depth"] == "basic":
        suggestions.append("Try a more detailed extraction using 'advanced' extract_depth")
    return suggestions
