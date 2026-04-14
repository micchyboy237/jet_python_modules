import asyncio
import shutil
from pathlib import Path
from typing import List

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


async def process_result(result):
    """Safe processing of each streamed result."""
    url = result.url

    if result.success:
        # Title is in metadata
        title = result.metadata.get("title") if result.metadata else None
        title = title or "N/A"

        # Safe markdown length
        if hasattr(result.markdown, "raw_markdown"):
            md_len = len(result.markdown.raw_markdown)
        else:
            md_len = len(str(result.markdown)) if result.markdown else 0

        extracted_len = (
            len(result.extracted_content)
            if getattr(result, "extracted_content", None)
            else 0
        )

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


async def crawl_streaming_example():
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

    # Browser settings
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
    )

    # Run config with streaming enabled
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        stream=True,  # Important: enables streaming
        delay_before_return_html=2.0,  # safe value
        # Optional extras you can enable:
        # js_code=None,
        # screenshot=False,
        # check_robots_txt=False,
    )

    # Updated CrawlerMonitor (compatible with current __init__)
    monitor = CrawlerMonitor(
        urls_total=len(urls),  # Helps the monitor show accurate progress
        refresh_rate=1.0,  # Refresh UI every 1 second
        enable_ui=True,  # Set False to disable terminal UI
        max_width=120,  # Maximum terminal width
    )

    # Memory-aware dispatcher
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=75.0,  # Pause crawling if memory exceeds 75%
        check_interval=1.0,
        max_session_permit=8,  # Max concurrent sessions
        memory_wait_timeout=300.0,
        rate_limiter=RateLimiter(
            base_delay=(1.0, 2.5),
            max_delay=30.0,
            max_retries=3,
        ),
        monitor=monitor,  # Attach the monitor
    )

    print("🚀 Starting **streaming** multi-URL crawl...\n")
    print("Live monitor will show progress in the terminal.\n")

    async with AsyncWebCrawler(config=browser_config) as crawler:
        # This is the streaming pattern
        async for result in await crawler.arun_many(
            urls=urls,
            config=run_config,
            dispatcher=dispatcher,
        ):
            await process_result(result)

    print("\n🎉 Streaming crawl finished successfully!")


if __name__ == "__main__":
    asyncio.run(crawl_streaming_example())
