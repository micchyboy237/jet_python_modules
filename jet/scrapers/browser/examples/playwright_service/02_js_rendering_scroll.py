"""
Demo 02: Async JS-rendered page with infinite scroll.

Target: quotes.toscrape.com/scroll — infinite-scroll variant that loads
new quotes via XHR as you scroll down.

Demonstrates:
  - Async context and scrape_url
  - Scroll strategy 'until_stable' with increment mode
  - wait_for_js for initial render delay
  - Retry logic on transient failures
  - Full-page screenshot after scrolling
"""

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from jet.scrapers.browser.playwright_service import (
    ScrapeResult,
    create_async_context,
    scrape_url,
)

TARGET_URL = "https://quotes.toscrape.com/scroll"


async def main() -> None:
    print("=" * 60)
    print("Demo 02: JS Rendering + Infinite Scroll (Async)")
    print(f"Target: {TARGET_URL}")
    print("=" * 60)

    context = await create_async_context(headless=True)
    try:
        print("\nScraping with scroll_strategy='until_stable'...")
        result: ScrapeResult = await scrape_url(
            context,
            TARGET_URL,
            scroll_strategy="until_stable",
            scroll_mode="increment",
            scroll_max_attempts=20,
            scroll_delay_ms=1200,
            scroll_timeout_ms=30000,
            wait_for_js=True,
            with_screenshot=True,
            max_retries=2,
            timeout=20000,
            use_cache=False,
        )

        print(f"\n  Status       : {result['status']}")
        print(f"  HTML length  : {len(result['html']) if result['html'] else 0} chars")
        print(
            f"  Screenshot   : "
            f"{len(result['screenshot']) if result['screenshot'] else 0} bytes"
        )
        print(f"  Dimensions   : {json.dumps(result['dimensions'], indent=4)}")

        if result["html"]:
            quote_count = result["html"].count('class="quote"')
            print(
                f"  Quotes found : {quote_count} "
                f"({'more than 10 = scroll worked ✓' if quote_count > 10 else '⚠ expected >10'})"
            )

        if result["screenshot"]:
            out_path = os.path.join(
                os.path.dirname(__file__), "02_scrolled_screenshot.png"
            )
            with open(out_path, "wb") as f:
                f.write(result["screenshot"])
            print(f"  Saved        : {out_path}")

        if result["status"] != "completed":
            print("\n  ⚠ Scrape did not complete. Check logs above.")

    finally:
        await context.close()
        print("\nContext closed. Demo complete.")


if __name__ == "__main__":
    asyncio.run(main())
