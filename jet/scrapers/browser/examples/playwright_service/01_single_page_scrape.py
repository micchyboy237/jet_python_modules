"""
Demo 01: Basic single-page synchronous scrape.

Target: books.toscrape.com — static HTML scraping test site.

Demonstrates:
  - Synchronous context creation and teardown
  - Single URL scrape with screenshot and dimensions
  - Redis caching (second run is instant)
  - Result inspection and screenshot saving
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from jet.scrapers.browser.playwright_service import (
    ScrapeResult,
    create_sync_context,
    scrape_url_sync,
)

TARGET_URL = "https://books.toscrape.com/catalogue/category/books/mystery_3/index.html"


def main() -> None:
    print("=" * 60)
    print("Demo 01: Single Page Synchronous Scrape")
    print(f"Target: {TARGET_URL}")
    print("=" * 60)

    context = create_sync_context(headless=True)
    try:
        # --- First scrape (hits network) ---
        print("\n[Run 1] Scraping from network...")
        result: ScrapeResult = scrape_url_sync(
            context,
            TARGET_URL,
            with_screenshot=True,
            use_cache=True,
            timeout=15000,
        )

        print(f"  Status     : {result['status']}")
        print(f"  HTML length: {len(result['html']) if result['html'] else 0} chars")
        print(
            f"  Screenshot : "
            f"{len(result['screenshot']) if result['screenshot'] else 0} bytes"
        )
        print(f"  Dimensions : {json.dumps(result['dimensions'], indent=4)}")

        if result["screenshot"]:
            out_path = os.path.join(os.path.dirname(__file__), "01_screenshot.png")
            with open(out_path, "wb") as f:
                f.write(result["screenshot"])
            print(f"  Saved      : {out_path}")

        # --- Second scrape (should hit cache) ---
        print("\n[Run 2] Scraping from cache...")
        cached_result: ScrapeResult = scrape_url_sync(
            context,
            TARGET_URL,
            with_screenshot=True,
            use_cache=True,
        )
        print(f"  Status     : {cached_result['status']}")
        html_match = result["html"] == cached_result["html"]
        print(f"  HTML match : {html_match}")
        print(f"  Cache hit confirmed ✓" if html_match else "  ✗ Mismatch!")

    finally:
        context.close()
        print("\nContext closed. Demo complete.")


if __name__ == "__main__":
    main()
