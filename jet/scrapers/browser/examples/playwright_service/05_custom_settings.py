"""
Demo 05: Custom ScraperSettings configuration.

Demonstrates:
  - Overriding output directory
  - Custom cache key prefix (project isolation)
  - Custom cache TTL
  - Custom scroll tuning parameters
  - Custom stealth script
  - Environment variable override (JET_SCRAPER_OUTPUT_DIR)
  - Settings passed explicitly to every function
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from jet.cache.redis.types import RedisConfigParams
from jet.scrapers.browser.playwright_service import (
    ScrapeResult,
    ScraperSettings,
    create_async_context,
    scrape_batch,
    scrape_url,
)

TARGET_URL = "https://quotes.toscrape.com/scroll"


async def main() -> None:
    print("=" * 60)
    print("Demo 05: Custom ScraperSettings")
    print("=" * 60)

    # Create project-specific settings
    custom_settings = ScraperSettings(
        output_dir=Path("/tmp/jet_scraper_demo_05"),
        context_subdir="demo_05_context",
        traces_subdir="demo_05_traces",
        redis_config=RedisConfigParams(port=6379),
        cache_ttl=1800,  # 30 min instead of default 1 hour
        cache_key_prefix="demo05",  # Isolated from other demos
        scroll_threshold_px=200,  # More lenient than default 150
        scroll_max_no_change=6,  # More tolerant than default 4
        stealth_script=(
            "delete window.__playwright;"
            "delete window.__pw_manual;"
            "console.log('[demo05] Custom stealth script active');"
        ),
    )

    print(f"\nOutput dir     : {custom_settings.output_dir}")
    print(f"Context dir    : {custom_settings.context_dir}")
    print(f"Traces dir     : {custom_settings.traces_dir}")
    print(f"Cache prefix   : {custom_settings.cache_key_prefix}")
    print(f"Cache TTL      : {custom_settings.cache_ttl}s")
    print(f"Scroll thresh  : {custom_settings.scroll_threshold_px}px")
    print(f"Scroll no-chg  : {custom_settings.scroll_max_no_change}")

    # Verify directories were created
    assert custom_settings.context_dir.exists(), "Context dir not created"
    assert custom_settings.traces_dir.exists(), "Traces dir not created"
    print("\nDirectories created ✓")

    # Use custom settings with single scrape
    print(f"\n--- Single scrape with custom settings ---")
    context = await create_async_context(headless=True, settings=custom_settings)
    try:
        result: ScrapeResult = await scrape_url(
            context,
            TARGET_URL,
            scroll_strategy="until_stable",
            scroll_mode="increment",
            scroll_max_attempts=10,
            scroll_delay_ms=1000,
            with_screenshot=False,
            use_cache=True,
            settings=custom_settings,
        )
        print(f"  Status: {result['status']}")
        print(f"  HTML  : {len(result['html']) if result['html'] else 0} chars")

        # Verify cache key uses custom prefix
        cache_key = custom_settings.cache_key(TARGET_URL)
        print(f"  Cache key: {cache_key}")
        assert cache_key.startswith("demo05:"), "Cache key prefix mismatch"
        print("  Cache key prefix verified ✓")

    finally:
        await context.close()

    # Use custom settings with batch scrape
    print(f"\n--- Batch scrape with custom settings ---")
    batch_urls = [
        "https://books.toscrape.com/catalogue/category/books/travel_2/index.html",
        "https://httpbin.org/html",
    ]
    count = 0
    async for result in scrape_batch(
        batch_urls,
        num_parallel=2,
        headless=True,
        show_progress=True,
        use_cache=True,
        scroll_strategy="none",
        with_screenshot=False,
        settings=custom_settings,
    ):
        count += 1
        print(f"  [{count}] {result['url'][:50]:<50} | {result['status']}")

    print(f"\nAll operations used custom settings ✓")
    print(f"Output isolated at: {custom_settings.output_dir}")
    print("\nDemo complete.")


if __name__ == "__main__":
    asyncio.run(main())
