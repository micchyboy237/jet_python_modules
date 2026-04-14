import asyncio
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

import aiofiles
from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CacheMode,
    CrawlerRunConfig,
)
from crawl4ai.async_dispatcher import (
    CrawlerMonitor,
    MemoryAdaptiveDispatcher,
    RateLimiter,
)
from crawl4ai.content_filter_strategy import BM25ContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_JSON = OUTPUT_DIR / "results.json"


async def init_json_file():
    if not RESULTS_JSON.exists():
        async with aiofiles.open(RESULTS_JSON, "w", encoding="utf-8") as f:
            await f.write("[]")


def calculate_relevance_score(result, user_query: str) -> float:
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


async def save_result_to_json(result, user_query: str) -> None:
    data: Dict[str, Any] = {
        "url": getattr(result, "url", None),
        "success": getattr(result, "success", False),
        "timestamp": asyncio.get_event_loop().time(),
        "query": user_query,
    }

    if getattr(result, "success", False):
        data["title"] = (
            result.metadata.get("title") if getattr(result, "metadata", None) else None
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
        data["error_message"] = getattr(result, "error_message", "Unknown error")
        data["status_code"] = getattr(result, "status_code", None)
        data["relevance_score"] = 0.0

    async with aiofiles.open(RESULTS_JSON, "r+", encoding="utf-8") as f:
        content = await f.read()
        results_list: List[Dict] = json.loads(content) if content.strip() else []
        results_list.append(data)

        results_list.sort(
            key=lambda x: (x.get("relevance_score", 0.0), x.get("timestamp", 0)),
            reverse=True,
        )

        await f.seek(0)
        await f.truncate()
        await f.write(json.dumps(results_list, indent=2, ensure_ascii=False))


async def process_result(result, user_query: str):
    url = getattr(result, "url", "Unknown")

    if getattr(result, "success", False):
        title = (
            result.metadata.get("title") if getattr(result, "metadata", None) else "N/A"
        )
        markdown_obj = getattr(result, "markdown", None)

        raw_len = len(getattr(markdown_obj, "raw_markdown", "") or "")
        fit_len = len(getattr(markdown_obj, "fit_markdown", "") or "")
        score = calculate_relevance_score(result, user_query)

        print(f"✅ [SUCCESS] {url}")
        print(f"   Title           : {title}")
        print(f"   Raw Markdown    : {raw_len:,} chars")
        if fit_len:
            print(f"   Fit Markdown    : {fit_len:,} chars (BM25 filtered)")
        print(f"   Relevance Score : {score:.3f} / 1.0")
        print("-" * 90)
    else:
        print(f"❌ [FAILED]  {url}")
        print(f"   Error : {getattr(result, 'error_message', 'Unknown error')}")
        if getattr(result, "status_code", None):
            print(f"   Status: {result.status_code}")
        print("-" * 90)

    await save_result_to_json(result, user_query)


async def crawl_streaming_example():
    await init_json_file()

    urls: List[str] = [
        "https://httpbin.org/html",
        "https://example.com",
        "https://www.python.org",
        "https://news.ycombinator.com",
        "https://github.com/unclecode/crawl4ai",
        "https://httpbin.org/json",
        "https://www.wikipedia.org",
    ]

    USER_QUERY = "AI web crawling and data extraction with Python"

    bm25_filter = BM25ContentFilter(
        user_query=USER_QUERY,
        bm25_threshold=1.0,
    )

    markdown_generator = DefaultMarkdownGenerator(content_filter=bm25_filter)

    # === KEY CHANGES HERE ===
    browser_config = BrowserConfig(
        headless=False,  # Set to True if you don't want to see tabs
        verbose=True,
    )

    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        stream=True,
        # Lower delay → faster parallel start
        delay_before_return_html=0.5,
        markdown_generator=markdown_generator,
        # This directly controls internal concurrency (very important for arun_many)
        semaphore_count=10,  # Increase if your machine can handle it
    )

    monitor = CrawlerMonitor(
        urls_total=len(urls),
        refresh_rate=1.0,
        enable_ui=True,
        max_width=130,
    )

    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=80.0,  # Be a bit more aggressive
        check_interval=1.0,
        max_session_permit=10,  # ← Increase this (main limiter for tabs/sessions)
        rate_limiter=RateLimiter(
            base_delay=(0.8, 2.0),  # Faster ramp-up
            max_delay=15.0,
            max_retries=2,
        ),
        monitor=monitor,
    )

    print("🚀 Starting streaming crawl with improved parallelism...")
    print(f'   Query : "{USER_QUERY}"')
    print(f"   Results → {RESULTS_JSON}\n")

    async with AsyncWebCrawler(config=browser_config) as crawler:
        async for result in await crawler.arun_many(
            urls=urls,
            config=run_config,
            dispatcher=dispatcher,
        ):
            await process_result(result, USER_QUERY)

    print(f"\n🎉 Crawl finished! Check {RESULTS_JSON}")


if __name__ == "__main__":
    asyncio.run(crawl_streaming_example())
