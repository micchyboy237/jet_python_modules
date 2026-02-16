from __future__ import annotations

import asyncio
import logging
from typing import Any, Literal, TypedDict

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from crawl4ai.async_configs import CacheMode
from rich.console import Console
from rich.logging import RichHandler

console = Console()
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, markup=True, rich_tracebacks=True)],
)
logger = logging.getLogger("crawl4ai_tool")


class CrawlResultDict(TypedDict):
    """Structured return type for crawl operations (RAG/LLM friendly)."""

    success: bool
    url: str
    error: str | None
    markdown: str | None
    cleaned_html: str | None
    content: str | None  # main extracted content per strategy
    links: list[str]
    media: list[dict[str, Any]]
    metadata: dict[str, Any]
    extract_strategy: str


class Crawl4AITool:
    """Reusable manager for Crawl4AI async crawler (sync interface for agents)."""

    def __init__(
        self,
        headless: bool = True,
        verbose: bool = False,
        cache_mode: CacheMode = CacheMode.BYPASS,
        exclude_external_links: bool = False,
        word_count_threshold: int | None = None,
    ):
        self.headless = headless
        self.verbose = verbose
        self.cache_mode = cache_mode
        self.exclude_external_links = exclude_external_links
        self.word_count_threshold = word_count_threshold
        self._crawler: AsyncWebCrawler | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def _ensure_crawler(self) -> AsyncWebCrawler:
        if self._crawler is None:
            browser_cfg = BrowserConfig(headless=self.headless, verbose=self.verbose)
            run_cfg = CrawlerRunConfig(
                cache_mode=self.cache_mode,
                exclude_external_links=self.exclude_external_links,
                word_count_threshold=self.word_count_threshold,
                verbose=self.verbose,
            )
            self._crawler = AsyncWebCrawler(config=browser_cfg, run_config=run_cfg)
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._crawler.awarmup())
            logger.info("[green]Crawl4AI crawler warmed up[/green]")
        return self._crawler

    def crawl(
        self,
        url: str,
        css_selector: str | None = None,
        wait_for: str | None = None,
        extract_strategy: Literal["markdown", "html", "text", "json"] = "markdown",
        **extra_run_kwargs: Any,
    ) -> CrawlResultDict:
        """
        Crawl a URL and return clean, structured data.

        Handles lifecycle internally. Returns dict suitable for LLM/RAG consumption.
        """
        crawler = self._ensure_crawler()

        async def _run() -> CrawlResultDict:
            try:
                result = await crawler.arun(
                    url=url,
                    css_selector=css_selector,
                    wait_for=wait_for,
                    **extra_run_kwargs,
                )
                if not result.success:
                    return {
                        "success": False,
                        "url": url,
                        "error": result.error_message or "Unknown crawl error",
                        "markdown": None,
                        "cleaned_html": None,
                        "content": None,
                        "links": [],
                        "media": [],
                        "metadata": {},
                        "extract_strategy": extract_strategy,
                    }

                content: str | None = None
                if extract_strategy == "markdown":
                    content = result.markdown
                elif extract_strategy == "html":
                    content = result.cleaned_html
                elif extract_strategy == "text":
                    content = result.extracted_content
                elif extract_strategy == "json":
                    content = result.extracted_content  # JSON mode assumed

                return {
                    "success": True,
                    "url": url,
                    "error": None,
                    "markdown": result.markdown,
                    "cleaned_html": result.cleaned_html,
                    "content": content,
                    "links": result.links.get("internal", [])
                    + result.links.get("external", []),
                    "media": result.media,
                    "metadata": result.metadata,
                    "extract_strategy": extract_strategy,
                }
            except Exception as e:
                logger.exception("Crawl failed for %s", url)
                return {
                    "success": False,
                    "url": url,
                    "error": str(e),
                    "markdown": None,
                    "cleaned_html": None,
                    "content": None,
                    "links": [],
                    "media": [],
                    "metadata": {},
                    "extract_strategy": extract_strategy,
                }

        return self._loop.run_until_complete(_run())

    def close(self) -> None:
        if self._crawler is not None and not self._crawler.closed:
            self._loop.run_until_complete(self._crawler.aclose())
            logger.info("[yellow]Crawl4AI crawler closed[/yellow]")
        if self._loop is not None and not self._loop.is_closed():
            self._loop.close()


# ──────────────────────────────────────────────────────────────────────────────
# Public Tool for smolagents (using @tool decorator – idiomatic & auto-schema)
# ──────────────────────────────────────────────────────────────────────────────

from smolagents import tool


@tool
def crawl_web_page(
    url: str,
    extract_as: Literal["markdown", "html", "text", "json"] = "markdown",
    css_selector: str | None = None,
    wait_for_selector: str | None = None,
) -> CrawlResultDict:
    """
    Crawl a webpage using Crawl4AI and return clean, structured content + metadata.

    This is the preferred tool for fetching web content in research / RAG tasks.

    Args:
        url: The full URL to crawl (e.g. "https://example.com/article").
        extract_as: Format of main content to return. Choose:
            - "markdown" (default): clean Markdown with headings/lists/tables (best for most LLMs).
            - "html": cleaned HTML without boilerplate.
            - "text": plain text only.
            - "json": structured extraction (requires schema in extra config).
        css_selector: Optional CSS selector to narrow extraction (e.g. "article.main", "#content").
        wait_for_selector: Optional selector to wait for before extraction (useful for JS-heavy pages).

    Returns:
        Dictionary with success flag, main content, links, metadata, etc.
        Use 'content' field for primary extracted text/markdown.
    """
    crawler = Crawl4AITool(headless=True, verbose=False)
    try:
        result = crawler.crawl(
            url=url,
            extract_strategy=extract_as,
            css_selector=css_selector,
            wait_for=wait_for_selector,
        )
        return result
    finally:
        crawler.close()
