"""
Demo 03: Concurrent batch scraping with progress tracking.

Targets: Mix of books.toscrape.com categories, httpbin.org endpoints,
and an intentionally invalid domain for error handling.

Demonstrates:
  - scrape_batch async generator (streaming results)
  - Concurrency control via num_parallel
  - Progress bar with tqdm
  - Cache integration across batch
  - Mixed success/failure handling
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from jet.scrapers.browser.playwright_service import (
    scrape_batch,
)

TEST_URLS = [
    "https://books.toscrape.com/catalogue/category/books/travel_2/index.html",
    "https://books.toscrape.com/catalogue/category/books/mystery_3/index.html",
    "https://books.toscrape.com/catalogue/category/books/historical-fiction_4/index.html",
    "https://books.toscrape.com/catalogue/category/books/sequential-art_5/index.html",
    "https://httpbin.org/html",
    "https://httpbin.org/xml",
    "https://quotes.toscrape.com/",
    "https://quotes.toscrape.com/tag/love/",
    "https://this-domain-does-not-exist-for-testing.invalid/page",
]


async def main() -> None:
    print("=" * 60)
    print("Demo 03: Batch Concurrent Scrape")
    print(f"URLs: {len(TEST_URLS)} | Parallel: 3 | Limit: None")
    print("=" * 60)

    stats = {"completed": 0, "failed": 0, "total_html_chars": 0}

    async for result in scrape_batch(
        TEST_URLS,
        num_parallel=3,
        headless=True,
        show_progress=True,
        use_cache=True,
        scroll_strategy="none",
        with_screenshot=False,
        timeout=15000,
        max_retries=1,
    ):
        status = result["status"]
        url_short = result["url"][:60]

        if status == "completed":
            html_len = len(result["html"]) if result["html"] else 0
            stats["completed"] += 1
            stats["total_html_chars"] += html_len
            print(f"  ✓ {url_short:<60} | {html_len:>6} chars")
        elif status == "started":
            continue
        else:
            stats["failed"] += 1
            print(f"  ✗ {url_short:<60} | {status}")

    print("\n" + "=" * 60)
    print("Batch Summary:")
    print(f"  Completed    : {stats['completed']}/{len(TEST_URLS)}")
    print(f"  Failed       : {stats['failed']}/{len(TEST_URLS)}")
    print(f"  Total HTML   : {stats['total_html_chars']:,} chars")
    avg = stats["total_html_chars"] // max(stats["completed"], 1)
    print(f"  Avg per page : {avg:,} chars")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
