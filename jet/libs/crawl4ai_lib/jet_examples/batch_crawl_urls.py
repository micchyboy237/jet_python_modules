import asyncio
import json
from typing import List

from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CacheMode,
    CrawlerMonitor,
    CrawlerRunConfig,
    DisplayMode,
    MemoryAdaptiveDispatcher,
    RateLimiter,
)


async def concurrent_crawl_example():
    # Sample URLs (replace with your own, e.g., from search results)
    urls: List[str] = [
        "https://docs.crawl4ai.com/",
        "https://github.com/unclecode/crawl4ai",
        "https://en.wikipedia.org/wiki/Web_crawler",
        "https://www.example.com",
        "https://httpbin.org/json",
        # Add more as needed
    ]

    print(f"🚀 Starting concurrent crawl of {len(urls)} URLs...\n")

    # 1. Browser configuration (global)
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        # proxy="http://your-proxy:port",  # Optional
    )

    # 2. Common run configuration
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,  # Fresh crawl
        stream=False,  # Change to True for streaming
        # extraction_strategy=...,        # Add JsonCssExtractionStrategy, LLMExtractionStrategy, etc.
        # markdown_generator=...,         # Add PruningContentFilter, etc.
        # js_code="window.scrollTo(0, document.body.scrollHeight);",  # For dynamic content
        # check_robots_txt=True,          # Respect robots.txt
    )

    # 3. Optional: Rate limiter (helps avoid detection/throttling)
    rate_limiter = RateLimiter(
        base_delay=(1.0, 3.0),  # Random delay between requests to same domain
        max_delay=30.0,
        max_retries=3,
        rate_limit_codes=[429, 503],
    )

    # 4. Optional: Monitor for live progress
    monitor = CrawlerMonitor(
        max_visible_rows=15,
        display_mode=DisplayMode.DETAILED,  # Or AGGREGATED
    )

    # 5. Choose a dispatcher
    # Option A: Memory-aware (best for large-scale or unstable environments)
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=75.0,  # Pause if memory > 75%
        check_interval=1.0,
        max_session_permit=8,  # Max concurrent browser sessions
        rate_limiter=rate_limiter,
        monitor=monitor,
    )

    # Option B: Fixed semaphore (uncomment to use instead)
    # dispatcher = SemaphoreDispatcher(
    #     max_session_permit=10,
    #     rate_limiter=rate_limiter,
    #     monitor=monitor,
    # )

    # 6. Perform concurrent crawl
    async with AsyncWebCrawler(config=browser_config) as crawler:
        if run_config.stream:
            # Streaming mode: process results as they finish
            results = []
            async for result in await crawler.arun_many(
                urls=urls,
                config=run_config,
                dispatcher=dispatcher,
            ):
                results.append(result)
                if result.success:
                    print(
                        f"✅ Completed: {result.url} | Markdown: ~{len(result.markdown.raw_markdown) if result.markdown else 0} chars"
                    )
                else:
                    print(f"❌ Failed: {result.url} | {result.error_message}")
        else:
            # Batch mode: wait for all
            results = await crawler.arun_many(
                urls=urls,
                config=run_config,
                dispatcher=dispatcher,
            )

            for result in results:
                if result.success:
                    print(f"✅ {result.url}")
                    # Access structured data, markdown, etc.
                    if hasattr(result, "dispatch_result") and result.dispatch_result:
                        dr = result.dispatch_result
                        print(
                            f"   Memory: {getattr(dr, 'memory_usage', 'N/A')} MB | Duration: {getattr(dr, 'end_time', 0) - getattr(dr, 'start_time', 0):.2f}s"
                        )
                else:
                    print(f"❌ {result.url} - {result.error_message}")

    # Save results
    output = [
        {
            "url": r.url,
            "success": r.success,
            "markdown_length": len(r.markdown.raw_markdown) if r.markdown else 0,
            "error": r.error_message,
        }
        for r in results
    ]

    with open("concurrent_crawl_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n🎉 Done! Processed {len(results)} URLs. Results saved to JSON.")


# Run the example
if __name__ == "__main__":
    asyncio.run(concurrent_crawl_example())
