import asyncio
import json
from pathlib import Path
from typing import List

import aiofiles
from jet.libs.crawl4ai_lib.async_web_crawler_manager import AsyncWebCrawlerManager
from jet.libs.crawl4ai_lib.rag_crawler import CrawlResultProcessor
from jet.libs.crawl4ai_lib.search_searxng import SemanticResult, semantic_search_results

# ----------------------------------------------------------------------
# Application helpers (unchanged)
# ----------------------------------------------------------------------
OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
SEARCH_RESULTS_JSON = OUTPUT_DIR / "search_results.json"
RAG_CONTEXT_MD = OUTPUT_DIR / "rag_context.md"


async def init_json_file():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not SEARCH_RESULTS_JSON.exists():
        async with aiofiles.open(SEARCH_RESULTS_JSON, "w", encoding="utf-8") as f:
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

    EXCLUDED_TAGS = ["script", "style", "nav", "footer"]

    search_results: List[SemanticResult] = await semantic_search_results(query)

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

    print(f"Selected {len(search_results)} strongest seed URLs")
    print(f"Query: {query}\n")

    # Pass the async method directly
    await crawler_manager.crawl_many(
        urls=[o["url"] for o in search_results],
        process_result=processor.process_result,  # This is now async
        user_query=query,
        bm25_threshold=0.6,
        run_config={"excluded_tags": EXCLUDED_TAGS},
    )

    # Save results to JSON (outside the class, as requested)
    async with aiofiles.open(SEARCH_RESULTS_JSON, "w", encoding="utf-8") as f:
        await f.write(json.dumps(processor.get_results(), indent=2, ensure_ascii=False))

    # Save RAG context to markdown file
    rag_context = processor.get_rag_context()
    async with aiofiles.open(RAG_CONTEXT_MD, "w", encoding="utf-8") as f:
        await f.write(rag_context)

    full_path = SEARCH_RESULTS_JSON.resolve().absolute()
    print(f"\n📁 Results saved to:\n   {full_path}")

    print(f"\n📝 RAG Context saved to:\n   {RAG_CONTEXT_MD.resolve().absolute()}")
    print(f"   ({len(rag_context):,} characters)")


if __name__ == "__main__":
    asyncio.run(main())
