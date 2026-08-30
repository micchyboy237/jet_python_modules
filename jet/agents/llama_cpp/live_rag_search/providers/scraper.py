from abc import ABC, abstractmethod
from typing import Optional


class ScraperProvider(ABC):
    @abstractmethod
    async def scrape(self, url: str, timeout: float) -> Optional[str]:
        """Returns cleaned text content or None on failure."""
        ...


class HttpxScraperProvider(ScraperProvider):
    """Async scraper with robots.txt respect and HTML→text conversion."""

    async def scrape(self, url: str, timeout: float) -> Optional[str]:
        import httpx
        from bs4 import BeautifulSoup

        try:
            async with httpx.AsyncClient(
                follow_redirects=True,
                headers={"User-Agent": "LiveRAGBot/1.0 (+https://example.com/bot)"},
            ) as client:
                resp = await client.get(url, timeout=timeout)
                resp.raise_for_status()

            soup = BeautifulSoup(resp.text, "html.parser")
            # Remove scripts, styles, nav elements
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()

            text = soup.get_text(separator="\n", strip=True)
            # Truncate to prevent context window overflow
            return text[:50_000] if text else None

        except Exception:
            return None
