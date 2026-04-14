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

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_JSON = OUTPUT_DIR / "results.json"


async def init_json_file():
    """Initialize the results.json as an empty array if it doesn't exist."""
    if not RESULTS_JSON.exists():
        async with aiofiles.open(RESULTS_JSON, "w", encoding="utf-8") as f:
            await f.write("[]")


async def save_result_to_json(result) -> None:
    """Append a single result to results.json in a thread-safe-ish way for streaming."""
    # Build a clean serializable dictionary
    data: Dict[str, Any] = {
        "url": result.url,
        "success": result.success,
        "timestamp": asyncio.get_event_loop().time(),  # or use datetime for ISO format
    }

    if result.success:
        data["title"] = result.metadata.get("title") if result.metadata else None
        data["status_code"] = getattr(result, "status_code", None)

        # Markdown
        if hasattr(result.markdown, "raw_markdown"):
            data["markdown"] = result.markdown.raw_markdown
            data["markdown_length"] = len(result.markdown.raw_markdown)
        else:
            md_str = str(result.markdown) if result.markdown else ""
            data["markdown"] = md_str
            data["markdown_length"] = len(md_str)

        # Extracted content (e.g. from extraction strategies)
        extracted = getattr(result, "extracted_content", None)
        if extracted:
            data["extracted_content"] = extracted
            data["extracted_length"] = (
                len(extracted) if isinstance(extracted, str) else 0
            )
    else:
        data["error_message"] = result.error_message or "Unknown error"
        data["status_code"] = getattr(result, "status_code", None)

    # Append to JSON array (read → modify → write)
    async with aiofiles.open(RESULTS_JSON, "r+", encoding="utf-8") as f:
        content = await f.read()
        results_list = json.loads(content) if content.strip() else []

        results_list.append(data)

        # Overwrite with updated list
        await f.seek(0)
        await f.truncate()
        await f.write(json.dumps(results_list, indent=2, ensure_ascii=False))


async def process_result(result):
    """Safe processing of each streamed result + save to JSON."""
    url = result.url

    # Print to console (keep your nice formatting)
    if result.success:
        title = result.metadata.get("title") if result.metadata else "N/A"
        if hasattr(result.markdown, "raw_markdown"):
            md_len = len(result.markdown.raw_markdown)
        else:
            md_len = len(str(result.markdown)) if result.markdown else 0

        extracted_len = len(getattr(result, "extracted_content", "") or "")

        print(f"✅ [SUCCESS] {url}")
        print(f"   Title : {title}")
        print(f"   Markdown length : {md_len:,} characters")
        if extracted_len > 0:
            print(f"   Extracted content : {extracted_len:,} characters")
        print("-" * 90)
    else:
        print(f"❌ [FAILED]  {url}")
        print(f"   Error : {result.error_message or 'Unknown error'}")
        if getattr(result, "status_code", None):
            print(f"   Status code : {result.status_code}")
        print("-" * 90)

    # Save to JSON (fire-and-forget style, but awaited for safety)
    await save_result_to_json(result)


async def crawl_streaming_example():
    await init_json_file()  # Ensure results.json starts as []

    # Example URLs
    urls: List[str] = [
        "https://httpbin.org/html",
        "https://example.com",
        "https://www.python.org",
        "https://news.ycombinator.com",
        "https://github.com/unclecode/crawl4ai",
        "https://httpbin.org/json",
        "https://www.wikipedia.org",
    ]

    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
    )

    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        stream=True,
        delay_before_return_html=2.0,
    )

    monitor = CrawlerMonitor(
        urls_total=len(urls),
        refresh_rate=1.0,
        enable_ui=True,
        max_width=120,
    )

    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=75.0,
        check_interval=1.0,
        max_session_permit=8,
        memory_wait_timeout=300.0,
        rate_limiter=RateLimiter(
            base_delay=(1.0, 2.5),
            max_delay=30.0,
            max_retries=3,
        ),
        monitor=monitor,
    )

    print("🚀 Starting **streaming** multi-URL crawl...\n")
    print("Results will be appended live to:")
    print(f"   {RESULTS_JSON}\n")

    async with AsyncWebCrawler(config=browser_config) as crawler:
        async for result in await crawler.arun_many(
            urls=urls,
            config=run_config,
            dispatcher=dispatcher,
        ):
            await process_result(result)

    print(f"\n🎉 Streaming crawl finished! Results saved to {RESULTS_JSON}")


if __name__ == "__main__":
    asyncio.run(crawl_streaming_example())
