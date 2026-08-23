"""
URL Context Extractor for ReAct Web Searcher.
Handles fetching via Playwright (for JS/Bot protection), parsing (via Trafilatura),
and truncating web content.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from jet.adapters.llama_cpp.chunking_utils import truncate_texts

logger = logging.getLogger(__name__)

READ_TIMEOUT = 30.0  # Increased for Playwright rendering
MAX_TOKENS_DEFAULT = 2048

# Generic User Agent to mimic a real browser
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


class UrlContextExtractor:
    """Extracts clean text content from a given URL using Playwright and Trafilatura."""

    def __init__(
        self,
        model: str = "qwen3.5-uncensored:2b",
        max_tokens: int = MAX_TOKENS_DEFAULT,
    ):
        self.model = model
        self.max_tokens = max_tokens

    async def extract(self, url: str, strict_sentences: bool = True) -> tuple[str, str]:
        """
        Fetch and extract content from a URL using Playwright.

        Args:
            url: The URL to fetch.
            strict_sentences: Whether to enforce sentence boundaries during truncation.

        Returns:
            A tuple of (clean_text, error_message).
        """
        logger.info("📄 Extracting context via Playwright: %s", url[:80])

        try:
            from playwright.async_api import async_playwright

            async with async_playwright() as p:
                # Launch headless browser
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(
                    user_agent=USER_AGENT,
                    viewport={"width": 1280, "height": 720},
                    ignore_https_errors=True,
                )
                page = await context.new_page()

                # Navigate and wait for network to be mostly idle
                try:
                    await page.goto(
                        url, wait_until="domcontentloaded", timeout=READ_TIMEOUT * 1000
                    )
                    # Additional wait for dynamic content (generic 2s wait or until network idle)
                    await page.wait_for_timeout(2000)
                except Exception as e:
                    return "", f"Navigation failed: {e}"

                # Get the full rendered HTML
                html_content = await page.content()
                await browser.close()

                if not html_content:
                    return "", "Page returned empty content."

                # Use Trafilatura for robust extraction from rendered HTML
                clean_text = self._extract_with_trafilatura(
                    html_content.encode("utf-8")
                )

                # Fallback to basic regex if Trafilatura fails
                if not clean_text:
                    logger.warning("Trafilatura failed, falling back to regex.")
                    clean_text = self._extract_with_regex(html_content)

                if not clean_text:
                    return "", "Page content is empty or could not be extracted."

                # Truncate to fit context window
                truncated = truncate_texts(
                    clean_text,
                    model=self.model,
                    max_tokens=self.max_tokens,
                    strict_sentences=strict_sentences,
                    show_progress=False,
                )

                final_text = (
                    truncated[0] if isinstance(truncated, list) and truncated else ""
                )

                if not final_text:
                    return "", "Content was too short or invalid after truncation."

                logger.info(
                    "✅ Extracted %d chars from %s",
                    len(final_text),
                    url[:60],
                )
                return final_text, ""

        except ImportError:
            return (
                "",
                "Playwright is not installed. Run 'pip install playwright' and 'playwright install'.",
            )
        except Exception as e:
            logger.error("❌ Failed to fetch/extract %s: %s", url[:60], e)
            return "", f"Failed to fetch URL: {e}"

    @staticmethod
    def _extract_with_trafilatura(content: bytes) -> Optional[str]:
        """Use Trafilatura for robust main-content extraction."""
        try:
            import trafilatura

            extracted = trafilatura.extract(
                content,
                include_links=False,
                include_tables=False,
                no_fallback=False,
            )
            return extracted.strip() if extracted else None
        except ImportError:
            logger.warning("Trafilatura not installed, skipping advanced extraction.")
            return None
        except Exception as e:
            logger.debug("Trafilatura extraction failed: %s", e)
            return None

    @staticmethod
    def _extract_with_regex(html: str) -> str:
        """Basic regex-based cleanup fallback."""
        clean = re.sub(r"<[^>]+>", " ", html)
        clean = re.sub(r"\s+", " ", clean).strip()
        return clean


if __name__ == "__main__":
    import asyncio
    import logging

    logging.basicConfig(level=logging.INFO)

    async def main():
        extractor = UrlContextExtractor(model="qwen3.5-uncensored:2b")

        # Test with a JS-heavy site
        test_url = "https://en.wikipedia.org/wiki/Isekai"

        print(f"Testing extraction for: {test_url}")
        content, error = await extractor.extract(test_url)

        if error:
            print(f"❌ Error: {error}")
        else:
            print(f"✅ Success! Extracted {len(content)} characters.")
            print("-" * 40)
            print(content[:500] + "...")
            print("-" * 40)

    asyncio.run(main())
