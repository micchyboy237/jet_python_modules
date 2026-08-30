# jet_python_modules/jet/agents/llama_cpp/live_rag_search/providers/scraper.py

from abc import ABC, abstractmethod
from typing import List, Optional
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from jet.logger import logger


class ScraperProvider(ABC):
    @abstractmethod
    async def scrape(self, url: str, timeout: float) -> Optional[str]:
        """Returns cleaned text content or None on failure."""
        ...

    @abstractmethod
    async def extract_links(self, html_content: str, base_url: str) -> List[str]:
        """Extracts absolute URLs from HTML content."""
        ...


class HttpxScraperProvider(ScraperProvider):
    """Lightweight async scraper using httpx. Good for static sites."""

    async def scrape(self, url: str, timeout: float) -> Optional[str]:
        import httpx

        try:
            async with httpx.AsyncClient(
                follow_redirects=True,
                headers={"User-Agent": "Mozilla/5.0 (compatible; LiveRAGBot/1.0)"},
                timeout=timeout,
            ) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                # Store raw HTML for link extraction if needed, but return text for LLM
                self._last_html = resp.text
                return self._clean_html(resp.text)
        except Exception as e:
            logger.warning(f"httpx scrape failed for {url}: {e}")
            return None

    async def extract_links(self, html_content: str, base_url: str) -> List[str]:
        # If html_content is actually text (cleaned), this might fail.
        # Ideally, we pass raw HTML. For now, we try to parse what we have.
        # In a real flow, agent.py should pass raw HTML to this method.
        soup = BeautifulSoup(html_content, "html.parser")
        links = []
        for tag in soup.find_all("a", href=True):
            href = tag["href"]
            if href.startswith(("http://", "https://")):
                links.append(href)
            elif href.startswith("/"):
                links.append(urljoin(base_url, href))
        return list(set(links))

    def _clean_html(self, html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)[:50_000]


class PlaywrightScraperProvider(ScraperProvider):
    """Robust scraper using Playwright. Handles JS rendering and scrolling."""

    async def scrape(self, url: str, timeout: float) -> Optional[str]:
        from playwright.async_api import async_playwright

        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context()
                page = await context.new_page()

                logger.debug(f"Playwright navigating to {url}")
                await page.goto(
                    url, timeout=timeout * 1000, wait_until="domcontentloaded"
                )
                await page.wait_for_timeout(2000)
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                await page.wait_for_timeout(1000)

                html_content = await page.content()
                await browser.close()

                self._last_html = html_content
                return self._clean_html(html_content)
        except Exception as e:
            logger.error(f"Playwright scrape failed for {url}: {e}")
            return None

    async def extract_links(self, html_content: str, base_url: str) -> List[str]:
        soup = BeautifulSoup(html_content, "html.parser")
        links = []
        for tag in soup.find_all("a", href=True):
            href = tag["href"]
            if href.startswith(("http://", "https://")):
                links.append(href)
            elif href.startswith("/"):
                links.append(urljoin(base_url, href))
        return list(set(links))

    def _clean_html(self, html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)[:50_000]
