"""Streaming RAG crawler that searches via SearXNG, crawls top results,
and generates structured RAG context.

Usage:
    # Run with default query and top 5 results
    python streaming_rag_crawler.py

    # Custom query with more results
    python streaming_rag_crawler.py "Python async web scraping best practices" -k 10

Outputs:
    search_results.json, rag_results.json, rag_context.md,
    and per-source markdown files
"""

import asyncio
import json
from pathlib import Path
from typing import List

import aiofiles
from jet.libs.crawl4ai_lib.async_web_crawler_manager import AsyncWebCrawlerManager
from jet.libs.crawl4ai_lib.config import (
    CRAWLER_MAX_SESSION_PERMIT,
    CRAWLER_MEMORY_THRESHOLD_PERCENT,
)
from jet.libs.crawl4ai_lib.rag_crawler import CrawlResultProcessor
from jet.libs.crawl4ai_lib.search_searxng import SemanticResult, semantic_search_results
from jet.logger import logger
from jet.utils.text import format_sub_source_dir

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
SEARCH_RESULTS_JSON = OUTPUT_DIR / "search_results.json"
RAG_RESULTS_JSON = OUTPUT_DIR / "rag_results.json"
RAG_CONTEXT_MD = OUTPUT_DIR / "rag_context.md"
RAG_RESULTS_DIR = OUTPUT_DIR / "rag"


async def init_json_file():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not SEARCH_RESULTS_JSON.exists():
        async with aiofiles.open(SEARCH_RESULTS_JSON, "w", encoding="utf-8") as f:
            await f.write("[]")


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
    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=5,
        help="Number of top search results to use (default: 5)",
    )
    args = parser.parse_args()

    query: str = args.query
    top_k: int = args.top_k

    EXCLUDED_TAGS = ["script", "style", "nav", "footer"]

    logger.info(f"Starting streaming RAG crawl: query='{query}', top_k={top_k}")

    search_results: List[SemanticResult] = await semantic_search_results(
        query, top_k=top_k
    )

    processor = CrawlResultProcessor()
    processor.set_current_query(query)

    # Manager creates MemoryAdaptiveDispatcher internally via _create_dispatcher()
    # using these constructor parameters for memory threshold and concurrency
    crawler_manager = AsyncWebCrawlerManager(
        headless=False,
        verbose=True,
        max_session_permit=CRAWLER_MAX_SESSION_PERMIT,
        memory_threshold_percent=CRAWLER_MEMORY_THRESHOLD_PERCENT,
    )

    print(f"Selected {len(search_results)} strongest seed URLs")
    print(f"Query: {query}\n")

    await crawler_manager.crawl_many(
        urls=[o["url"] for o in search_results],
        process_result=processor.process_result,
        user_query=query,
        bm25_threshold=0.6,
        run_config={
            "excluded_tags": EXCLUDED_TAGS,
            "word_count_threshold": 50,
        },
    )

    # Save search results
    async with aiofiles.open(SEARCH_RESULTS_JSON, "w", encoding="utf-8") as f:
        await f.write(json.dumps(processor.get_results(), indent=2, ensure_ascii=False))

    # Generate and save RAG context
    rag_context = processor.get_rag_context()
    async with aiofiles.open(RAG_CONTEXT_MD, "w", encoding="utf-8") as f:
        await f.write(rag_context)

    # Save clean RAG results (without large markdown blobs)
    rag_results = processor.get_results()
    clean_results = []
    for o in rag_results:
        o_copy = o.copy()
        o_copy.pop("raw_markdown", None)
        o_copy.pop("fit_markdown", None)
        o_copy.pop("markdown", None)
        clean_results.append(o_copy)

    async with aiofiles.open(RAG_RESULTS_JSON, "w", encoding="utf-8") as f:
        await f.write(json.dumps(clean_results, indent=2, ensure_ascii=False))

    # Save per-source files
    for o in rag_results:
        sub_dir = RAG_RESULTS_DIR / format_sub_source_dir(o["url"])
        sub_dir.mkdir(parents=True, exist_ok=True)

        o_copy = o.copy()
        raw_markdown = o_copy.pop("raw_markdown", "")
        fit_markdown = o_copy.pop("fit_markdown", "")
        o_copy.pop("markdown", None)

        raw_md_file = sub_dir / "raw_markdown.md"
        fit_md_file = sub_dir / "fit_markdown.md"
        result_file = sub_dir / "result.json"

        async with aiofiles.open(raw_md_file, "w", encoding="utf-8") as f:
            await f.write(raw_markdown)
        async with aiofiles.open(fit_md_file, "w", encoding="utf-8") as f:
            await f.write(fit_markdown)
        async with aiofiles.open(result_file, "w", encoding="utf-8") as f:
            await f.write(json.dumps(o_copy, indent=2, ensure_ascii=False))

    full_path = SEARCH_RESULTS_JSON.resolve().absolute()
    print(f"\n📁 Search results saved to:\n   {full_path}")

    full_path = RAG_RESULTS_JSON.resolve().absolute()
    print(f"\n📁 RAG results saved to:\n   {full_path}")

    print(f"\n📝 RAG Context saved to:\n   {RAG_CONTEXT_MD.resolve().absolute()}")
    print(f"   ({len(rag_context):,} characters)")

    logger.info("Streaming RAG crawl completed successfully")


if __name__ == "__main__":
    asyncio.run(main())
