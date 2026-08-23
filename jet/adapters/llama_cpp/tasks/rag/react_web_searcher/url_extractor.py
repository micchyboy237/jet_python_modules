"""
URL Context Extractor for ReAct Web Searcher.
Uses jet.scrapers.playwright_utils for robust browser management and
SmartChunker + hybrid_search for query-aware content extraction.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from jet.adapters.llama_cpp.chunk_strategies import (
    SmartChunker,
    estimate_tokens_safe,
    format_chunks_for_rag,
)
from jet.adapters.llama_cpp.hybrid_utils import hybrid_search
from jet.scrapers.playwright_utils import scrape_urls

logger = logging.getLogger(__name__)

MAX_TOKENS_DEFAULT = 2048
MIN_VALID_TOKENS = 64


class UrlContextExtractor:
    """Extracts clean text content from a URL using playwright_utils + SmartChunker."""

    def __init__(
        self,
        model: str = "qwen3.5-uncensored:2b",
        max_tokens: int = MAX_TOKENS_DEFAULT,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self._chunker = SmartChunker(model)

    async def extract(
        self,
        url: str,
        strict_sentences: bool = True,
        query: str | None = None,  # ✅ NEW: Optional query for targeted extraction
    ) -> tuple[str, str]:
        """
        Fetch and extract content from a URL.

        Args:
            url: The URL to fetch.
            strict_sentences: Unused (kept for API compat).
            query: Optional search query. If provided, uses hybrid_search
                   to return only the most relevant chunks instead of raw top chunks.
        """
        logger.info(
            "📄 Extracting context via playwright_utils: %s (query=%r)",
            url[:80],
            query[:60] if query else None,
        )

        try:
            html_content = ""
            async for result in scrape_urls(
                urls=[url],
                num_parallel=1,
                limit=1,
                show_progress=False,
                timeout=15000,
                max_retries=2,
                with_screenshot=False,
                headless=True,
                wait_for_js=False,
                use_cache=True,
                scroll_strategy="until_stable",
                scroll_mode="increment",
                scroll_max_attempts=10,
            ):
                if result["status"] == "completed" and result["html"]:
                    html_content = result["html"]
                    logger.info(
                        "✅ playwright_utils fetched %d chars from %s",
                        len(html_content),
                        url[:60],
                    )
                elif result["status"] in ("failed_no_html", "failed_error"):
                    logger.warning(
                        "⚠️ playwright_utils failed for %s: status=%s",
                        url[:60],
                        result["status"],
                    )

            if not html_content:
                return "", "Page returned empty content or fetch failed."

            clean_text = self._extract_with_trafilatura(html_content.encode("utf-8"))
            if not clean_text:
                logger.warning("Trafilatura failed, falling back to regex.")
                clean_text = self._extract_with_regex(html_content)

            if not clean_text:
                return "", "Page content is empty or could not be extracted."

            total_tokens = estimate_tokens_safe(clean_text, model=self.model)
            logger.info(
                "📊 Raw extraction: %d chars, ~%d tokens (budget=%d)",
                len(clean_text),
                total_tokens,
                self.max_tokens,
            )

            # If content fits entirely, return as-is regardless of query
            if total_tokens <= self.max_tokens:
                final_text = clean_text.strip()
            else:
                # Chunk the content first
                chunks = self._chunker.chunk(
                    text=clean_text,
                    chunk_size=min(self.max_tokens, 512),
                    chunk_overlap=0,
                    min_chunk_size=MIN_VALID_TOKENS,
                    buffer=4,
                    retrieval_type="dense",
                )

                if not chunks:
                    return "", "SmartChunker produced no valid chunks."

                # ✅ NEW: Query-aware selection via hybrid_search
                if query:
                    logger.info(
                        "🔍 Applying hybrid_search for query-aware extraction (%d chunks)",
                        len(chunks),
                    )
                    try:
                        # Retrieve top results fitting within token budget
                        # We retrieve more than needed to allow for token assembly
                        search_results = hybrid_search(
                            query=query,
                            documents=chunks,
                            top_n=min(len(chunks), 10),
                            normalize_scores=True,
                        )
                        # Use ranked chunks instead of raw sequential chunks
                        selected_chunks = [r["text"] for r in search_results]
                        logger.info(
                            "✅ Hybrid search selected %d relevant chunks",
                            len(selected_chunks),
                        )
                    except Exception as e:
                        logger.warning(
                            "Hybrid search failed (%s), falling back to sequential chunks",
                            e,
                        )
                        selected_chunks = chunks
                else:
                    selected_chunks = chunks

                formatted = format_chunks_for_rag(selected_chunks)

                # Assemble within token budget
                assembled_parts: list[str] = []
                assembled_tokens = 0
                for chunk in formatted:
                    chunk_tokens = estimate_tokens_safe(chunk, model=self.model)
                    if assembled_tokens + chunk_tokens > self.max_tokens:
                        break
                    assembled_parts.append(chunk)
                    assembled_tokens += chunk_tokens

                final_text = "\n---\n".join(assembled_parts).strip()
                logger.info(
                    "✅ Final assembly: %d chunks, %d/%d tokens used",
                    len(assembled_parts),
                    assembled_tokens,
                    self.max_tokens,
                )

            final_tokens = estimate_tokens_safe(final_text, model=self.model)
            if final_tokens < MIN_VALID_TOKENS:
                return (
                    "",
                    f"Content too short after processing: {final_tokens} tokens (minimum {MIN_VALID_TOKENS})",
                )

            logger.info(
                "✅ Final: %d chars (%d tokens) from %s",
                len(final_text),
                final_tokens,
                url[:60],
            )
            return final_text, ""

        except ImportError as e:
            return "", f"Missing dependency: {e}"
        except Exception as e:
            logger.error("❌ Failed to fetch/extract %s: %s", url[:60], e)
            return "", f"Failed to fetch URL: {e}"

    @staticmethod
    def _extract_with_trafilatura(content: bytes) -> Optional[str]:
        """Use Trafilatura for robust main-content extraction.

        ✅ Tables enabled + markdown output for SmartChunker compatibility.
        """
        try:
            import trafilatura

            extracted = trafilatura.extract(
                content,
                include_links=False,
                include_tables=True,
                output_format="markdown",
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
        test_urls = [
            "https://en.wikipedia.org/wiki/Isekai",
            "https://myanimelist.net/stacks/28254",
            "https://httpbin.org/status/404",
        ]
        print("=" * 60)
        print("URL CONTEXT EXTRACTION DEMO (playwright_utils)")
        print("=" * 60)
        for url in test_urls:
            print(f"\n{'-' * 60}")
            print(f"🔍 Testing: {url}")
            print(f"{'-' * 60}")
            content, error = await extractor.extract(url)
            if error:
                print(f"❌ Error: {error}")
            else:
                print(f"✅ Success! {len(content)} chars")
                print(content[:300] + "...")

    asyncio.run(main())
