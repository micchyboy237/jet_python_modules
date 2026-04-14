import asyncio
import json
from pathlib import Path
from typing import Any, List

import aiofiles
from jet.libs.crawl4ai_lib.async_web_crawler_manager import AsyncWebCrawlerManager
from jet.libs.crawl4ai_lib.search_searxng import semantic_search_results

# ----------------------------------------------------------------------
# Application helpers (unchanged)
# ----------------------------------------------------------------------
OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
RESULTS_JSON = OUTPUT_DIR / "results.json"
RAG_CONTEXT_MD = OUTPUT_DIR / "rag_context.md"


# ----------------------------------------------------------------------
# Reusable Result Processor Class
# ----------------------------------------------------------------------
class CrawlResultProcessor:
    """Reusable class for processing crawl results with RAG support."""

    def __init__(self):
        self.results: List[dict[str, Any]] = []
        self._current_user_query: str = ""

    def set_current_query(self, query: str) -> None:
        """Set the current query so it can be used during processing."""
        self._current_user_query = query

    async def process_result(self, result: Any) -> None:
        """
        Async callback required by crawl_many.
        This version is now properly awaitable.
        """
        try:
            url = getattr(result, "url", "Unknown")
            success = getattr(result, "success", False)
            user_query = self._current_user_query

            if success:
                title = (
                    result.metadata.get("title")
                    if getattr(result, "metadata", None)
                    else "N/A"
                )
                markdown_obj = getattr(result, "markdown", None)
                raw_len = len(getattr(markdown_obj, "raw_markdown", "") or "")
                fit_len = len(getattr(markdown_obj, "fit_markdown", "") or "")
                score = (
                    calculate_relevance_score(result, user_query) if user_query else 0.0
                )

                print(f"✅ [SUCCESS] {url}")
                print(f" Title : {title}")
                print(f" Raw Markdown : {raw_len:,} chars")
                if fit_len:
                    print(f" Fit Markdown : {fit_len:,} chars (BM25 filtered)")
                print(f" Relevance Score : {score:.3f} / 1.0")
                print("-" * 90)
            else:
                error_msg = getattr(result, "error_message", "Unknown error")
                status = getattr(result, "status_code", None)
                print(f"❌ [FAILED] {url}")
                print(f" Error : {error_msg}")
                if status:
                    print(f" Status: {status}")
                print("-" * 90)

            # Store result (same structure as before)
            data: dict[str, Any] = {
                "url": getattr(result, "url", None),
                "success": success,
                "timestamp": asyncio.get_event_loop().time(),
                "query": user_query,
            }

            if success:
                data["title"] = (
                    result.metadata.get("title")
                    if getattr(result, "metadata", None)
                    else None
                )
                data["status_code"] = getattr(result, "status_code", None)

                markdown_obj = getattr(result, "markdown", None)
                if hasattr(markdown_obj, "raw_markdown"):
                    data["raw_markdown"] = markdown_obj.raw_markdown
                    data["raw_markdown_length"] = len(markdown_obj.raw_markdown)

                fit_md = getattr(markdown_obj, "fit_markdown", None)
                if fit_md:
                    data["fit_markdown"] = fit_md
                    data["fit_markdown_length"] = len(fit_md)
                    data["markdown"] = fit_md
                else:
                    md_str = getattr(markdown_obj, "raw_markdown", None) or str(
                        markdown_obj or ""
                    )
                    data["markdown"] = md_str
                    data["markdown_length"] = len(md_str)

                data["relevance_score"] = calculate_relevance_score(result, user_query)

                extracted = getattr(result, "extracted_content", None)
                if extracted:
                    data["extracted_content"] = extracted
                    data["extracted_length"] = (
                        len(extracted) if isinstance(extracted, str) else 0
                    )
            else:
                data["error_message"] = getattr(
                    result, "error_message", "Unknown error"
                )
                data["status_code"] = getattr(result, "status_code", None)
                data["relevance_score"] = 0.0

            self.results.append(data)

            # Keep results sorted by relevance score (descending)
            self.results.sort(
                key=lambda x: (x.get("relevance_score", 0.0), x.get("timestamp", 0)),
                reverse=True,
            )

        except Exception as e:
            print(
                f"⚠️ Error processing result for {getattr(result, 'url', 'Unknown')}: {e}"
            )
            self.results.append(
                {
                    "url": getattr(result, "url", None),
                    "success": False,
                    "error_message": str(e),
                    "relevance_score": 0.0,
                    "timestamp": asyncio.get_event_loop().time(),
                }
            )

    def get_rag_context(self) -> str:
        """Generate clean RAG-ready context from successful results."""
        if not self.results:
            return ""

        parts: List[str] = []
        for item in self.results:
            if not item.get("success", False):
                continue

            url = item.get("url", "")
            title = item.get("title", "Untitled")
            markdown = (
                item.get("markdown")
                or item.get("fit_markdown")
                or item.get("raw_markdown", "")
            ).strip()

            if markdown and len(markdown) > 50:
                header = f"Source: {title}\nURL: {url}\n\n"
                parts.append(header + markdown + "\n\n" + "─" * 80 + "\n\n")

        return "".join(parts).strip()

    def get_results(self) -> List[dict[str, Any]]:
        return self.results.copy()

    def clear(self) -> None:
        self.results.clear()


def calculate_relevance_score(result: Any, user_query: str) -> float:
    """Original relevance calculation — unchanged."""
    if not getattr(result, "success", False):
        return 0.0

    markdown_obj = getattr(result, "markdown", None)
    if not markdown_obj:
        return 0.0

    fit_md = getattr(markdown_obj, "fit_markdown", None)
    raw_md = getattr(markdown_obj, "raw_markdown", None) or str(markdown_obj)
    content = fit_md or raw_md

    if not content or len(content) < 50:
        return 0.0

    ratio = len(content) / max(len(raw_md), 1)
    survival_score = min(1.0, ratio * 1.8)

    query_terms = [term.lower() for term in user_query.split() if len(term) > 2]
    if not query_terms:
        return round(survival_score, 3)

    text_lower = content.lower()
    hits = sum(text_lower.count(term) for term in query_terms)
    density = hits / max(len(content.split()), 1)
    keyword_bonus = min(0.6, density * 8)

    final_score = (survival_score * 0.65) + (keyword_bonus * 0.35)
    return round(min(1.0, final_score), 3)


async def init_json_file():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not RESULTS_JSON.exists():
        async with aiofiles.open(RESULTS_JSON, "w", encoding="utf-8") as f:
            await f.write("[]")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
async def main():
    await init_json_file()

    import argparse

    parser = argparse.ArgumentParser(description="Streaming RAG Crawler")
    parser.add_argument(
        "query",
        nargs="?",
        default="AI web crawling and data extraction with Python",
        help="Search query for crawling",
    )
    args = parser.parse_args()
    query: str = args.query

    urls: List[str] = await semantic_search_results(query)

    # Initialize processor
    processor = CrawlResultProcessor()
    processor.set_current_query(query)

    crawler_manager = AsyncWebCrawlerManager(
        headless=False,
        verbose=True,
        max_session_permit=8,  # Lowered for stability
        semaphore_count=10,
        memory_threshold_percent=75.0,
        base_delay=(1.2, 3.5),  # Gentler delays
        delay_before_return_html=1.0,
        monitor_max_width=130,
    )

    print(f"Selected {len(urls)} strongest seed URLs")
    print(f"Query: {query}\n")

    # Pass the async method directly
    await crawler_manager.crawl_many(
        urls=urls,
        process_result=processor.process_result,  # This is now async
        user_query=query,
        bm25_threshold=0.6,
    )

    # Save results to JSON (outside the class, as requested)
    async with aiofiles.open(RESULTS_JSON, "w", encoding="utf-8") as f:
        await f.write(json.dumps(processor.get_results(), indent=2, ensure_ascii=False))

    # Save RAG context to markdown file
    rag_context = processor.get_rag_context()
    async with aiofiles.open(RAG_CONTEXT_MD, "w", encoding="utf-8") as f:
        await f.write(rag_context)

    full_path = RESULTS_JSON.resolve().absolute()
    print(f"\n📁 Results saved to:\n   {full_path}")

    print(f"\n📝 RAG Context saved to:\n   {RAG_CONTEXT_MD.resolve().absolute()}")
    print(f"   ({len(rag_context):,} characters)")


if __name__ == "__main__":
    asyncio.run(main())
