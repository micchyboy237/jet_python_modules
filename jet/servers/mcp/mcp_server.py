from urllib.parse import urlparse, urljoin
from starlette.responses import JSONResponse
from starlette.requests import Request
from typing import AsyncIterator, List, Dict, Optional, TypedDict
from contextlib import asynccontextmanager
from mcp.server.fastmcp.server import FastMCP, Context
from pydantic import BaseModel, Field
from typing import Literal
from fake_useragent import UserAgent
from playwright.async_api import async_playwright

PLAYWRIGHT_CHROMIUM_EXECUTABLE = "/Users/jethroestrada/Library/Caches/ms-playwright/chromium-1187/chrome-mac/Chromium.app/Contents/MacOS/Chromium"


class FileInput(BaseModel):
    file_path: str = Field(...,
                           description="Path to the file (e.g., 'example.txt')")
    encoding: Literal["utf-8",
                      "ascii"] = Field("utf-8", description="File encoding")


class FileOutput(BaseModel):
    text: str = Field(..., description="File contents or error message")


class UrlInput(BaseModel):
    url: str = Field(..., description="URL to navigate to (e.g., 'https://example.com')",
                     pattern=r"^https?://")


class UrlOutput(BaseModel):
    url: str = Field(..., description="The URL that was navigated to")
    title: Optional[str] = Field(
        None, description="Page title or error message")
    nav_links: Optional[List[str]] = Field(
        None, description="List of links from the same server")
    text: Optional[str] = Field(
        None, description="All visible text content on the page")


class SummarizeTextInput(BaseModel):
    text: str = Field(..., description="Text to summarize")
    max_words: int = Field(
        100, description="Maximum number of words for the summary", ge=10, le=500)


class SummarizeTextOutput(BaseModel):
    word_count: int = Field(..., description="Number of words in the summary")
    text: str = Field(..., description="Summarized text")


class SearchTextsInput(BaseModel):
    texts: List[str] = Field(...,
                             description="List of text strings to search through")
    query: str = Field(..., description="Search query string")
    text_ids: Optional[List[str]] = Field(
        None, description="Optional identifiers for each text")
    top_k: Optional[int] = Field(
        None, description="Maximum number of results to return", ge=1)
    threshold: float = Field(
        0.0, description="Minimum similarity score for results", ge=0.0, le=1.0)
    chunk_size: int = Field(
        500, description="Size of text chunks", ge=50, le=1000)
    chunk_overlap: int = Field(
        100, description="Overlap between chunks", ge=0, le=200)
    split_chunks: bool = Field(
        False, description="Return individual chunks if True, merge if False")


class TextSearchMetadata(TypedDict):
    """Typed dictionary for search result metadata."""
    text_id: str
    start_idx: int
    end_idx: int
    chunk_idx: int
    content_similarity: float
    num_tokens: int


class TextSearchResult(TypedDict):
    """Typed dictionary for search result structure."""
    rank: int
    score: float
    metadata: TextSearchMetadata
    text: str


class SearchTextsOutput(BaseModel):
    results: List[TextSearchResult] = Field(
        ..., description="List of search results with metadata")
    error: Optional[str] = Field(
        None, description="Error message if search failed")


@asynccontextmanager
async def lifespan(app: FastMCP[None]) -> AsyncIterator[None]:
    print("Starting FastMCP server...")
    yield
    print("Shutting down FastMCP server...")

server = FastMCP(
    name="FastMCPStandalone",
    instructions="A standalone MCP server with file, browser, and text search tools.",
    debug=True,
    log_level="DEBUG",
    lifespan=lifespan
)


@server.tool(description="Read the contents of a file.", annotations={"audience": ["user"], "priority": 0.9})
async def read_file(arguments: FileInput, ctx: Context) -> FileOutput:
    await ctx.info(f"Reading file: {arguments.file_path}")
    try:
        with open(arguments.file_path, "r", encoding=arguments.encoding) as f:
            content = f.read()
        await ctx.report_progress(100, 100, "File read successfully")
        return FileOutput(text=content)
    except Exception as e:
        await ctx.error(f"Error reading file: {str(e)}")
        return FileOutput(text=f"Error reading file: {str(e)}")


@server.tool(description="Navigate to a URL and return the page title, links from the same server, and all visible text content.", annotations={"audience": ["assistant"], "priority": 0.8})
async def navigate_to_url(arguments: UrlInput, ctx: Context) -> UrlOutput:
    await ctx.info(f"Navigating to {arguments.url}")
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=False,
                executable_path=PLAYWRIGHT_CHROMIUM_EXECUTABLE,
                timeout=10000,
            )
            ua = UserAgent()
            page = await browser.new_page(user_agent=ua.random)
            await page.goto(arguments.url, wait_until="domcontentloaded", timeout=10000)
            await page.wait_for_function(
                '''() => {
                    return document.body.innerText.trim().length > 0;
                }''',
                timeout=10000
            )
            title = await page.title()
            link_elements = await page.query_selector_all('a[href]')
            seen_links = set()
            links = []
            parsed_url = urlparse(arguments.url)
            base_domain = parsed_url.netloc
            for element in link_elements:
                href = await element.get_attribute('href')
                if href:
                    absolute_url = urljoin(arguments.url, href)
                    parsed_link = urlparse(absolute_url)
                    if not parsed_link.netloc or parsed_link.netloc == base_domain:
                        if absolute_url not in seen_links:
                            seen_links.add(absolute_url)
                            links.append(absolute_url)
            text_content = await page.evaluate('''() => {
                return document.body.innerText.trim();
            }''')
            await browser.close()
        await ctx.report_progress(100, 100, "Navigation complete")
        return UrlOutput(
            url=arguments.url,
            title=title,
            nav_links=links or None,
            text=text_content or None
        )
    except Exception as e:
        await ctx.error(f"Error navigating to {arguments.url}: {str(e)}")
        return UrlOutput(
            url=arguments.url,
            title=None,
            nav_links=None,
            text=f"Error navigating to {arguments.url}: {str(e)}"
        )


@server.tool(description="Summarize text content to a specified word limit.", annotations={"audience": ["assistant"], "priority": 0.7})
async def summarize_text(arguments: SummarizeTextInput, ctx: Context) -> SummarizeTextOutput:
    await ctx.info(f"Summarizing text (max {arguments.max_words} words)")
    try:
        words = arguments.text.split()
        summary_words = words[:arguments.max_words]
        summary = " ".join(summary_words)
        if len(words) > arguments.max_words:
            summary += "..."
        word_count = len(summary_words)
        await ctx.report_progress(100, 100, "Summary generated")
        return SummarizeTextOutput(text=summary, word_count=word_count)
    except Exception as e:
        await ctx.error(f"Error summarizing text: {str(e)}")
        return SummarizeTextOutput(text=f"Error: {str(e)}", word_count=0)


@server.tool(description="Search through a list of texts using semantic similarity.", annotations={"audience": ["assistant"], "priority": 0.75})
async def search_texts_tool(arguments: SearchTextsInput, ctx: Context) -> SearchTextsOutput:
    from jet.vectors.semantic_search.text_vector_search import search_texts

    await ctx.info(f"Searching texts with query: {arguments.query}")
    try:
        results_iter = search_texts(
            texts=arguments.texts,
            query=arguments.query,
            text_ids=arguments.text_ids,
            top_k=arguments.top_k,
            threshold=arguments.threshold,
            chunk_size=arguments.chunk_size,
            chunk_overlap=arguments.chunk_overlap,
            split_chunks=arguments.split_chunks
        )
        results = list(results_iter)
        await ctx.report_progress(100, 100, f"Found {len(results)} matching text chunks")
        return SearchTextsOutput(results=results)
    except Exception as e:
        await ctx.error(f"Error searching texts: {str(e)}")
        return SearchTextsOutput(results=[], error=f"Error searching texts: {str(e)}")


@server.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


@server.resource("resource://welcome", description="A welcome message")
async def welcome_message() -> str:
    return "Welcome to FastMCP!"


@server.prompt(description="Analyze a file")
async def analyze_file(path: str) -> List[Dict]:
    content = open(path, "r").read()
    return [{"role": "user", "content": f"Analyze this content:\n{content}"}]

if __name__ == "__main__":
    server.run(transport="stdio")
