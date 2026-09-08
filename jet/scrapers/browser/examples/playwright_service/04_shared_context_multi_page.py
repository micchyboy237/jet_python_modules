"""
Demo 04: Shared context with multiple pages and manual navigation.

Target: books.toscrape.com — navigate through paginated category pages
within a single persistent context to demonstrate cookie/session sharing,
manual page control, and explicit resource cleanup.

Demonstrates:
  - Creating ONE context and opening MULTIPLE pages from it
  - Manual page.goto() + page.content() without scrape_url wrapper
  - Cookies persist across pages (same session)
  - Explicit page.close() for each page
  - Context reuse avoids repeated browser launch overhead
  - Works with both stealth patching and dynamic config
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from jet.scrapers.browser.playwright_service import (
    PageDimensions,
    create_async_context,
)

CATEGORY_PAGES = [
    "https://books.toscrape.com/catalogue/category/books/mystery_3/index.html",
    "https://books.toscrape.com/catalogue/category/books/mystery_3/page-2.html",
    "https://books.toscrape.com/catalogue/category/books/mystery_3/page-3.html",
    "https://books.toscrape.com/catalogue/category/books/mystery_3/page-4.html",
]


async def main() -> None:
    print("=" * 60)
    print("Demo 04: Shared Context Multi-Page Navigation")
    print(f"Pages: {len(CATEGORY_PAGES)}")
    print("=" * 60)

    # Single context shared across all pages
    context = await create_async_context(headless=True)
    try:
        # Verify cookies are empty at start
        initial_cookies = await context.cookies()
        print(f"\nInitial cookies: {len(initial_cookies)}")

        all_titles: list[str] = []

        for i, url in enumerate(CATEGORY_PAGES, 1):
            page = await context.new_page()
            try:
                print(f"\n[Page {i}/{len(CATEGORY_PAGES)}] {url}")
                await page.goto(url, wait_until="domcontentloaded", timeout=15000)

                # Extract book titles from this page
                titles = await page.evaluate("""() => {
                    const articles = document.querySelectorAll('article.product_pod h3 a');
                    return Array.from(articles).map(a => a.getAttribute('title'));
                }""")
                all_titles.extend(titles)
                print(f"  Books found: {len(titles)}")
                for title in titles[:3]:
                    print(f"    - {title}")
                if len(titles) > 3:
                    print(f"    ... and {len(titles) - 3} more")

                # Get dimensions to confirm page rendered
                dims: PageDimensions = await page.evaluate("""() => ({
                    width: document.documentElement.clientWidth,
                    height: document.documentElement.clientHeight,
                    deviceScaleFactor: window.devicePixelRatio
                })""")
                print(f"  Viewport: {dims['width']}x{dims['height']}")

            finally:
                await page.close()

        # Verify cookies accumulated across pages (session persistence)
        final_cookies = await context.cookies()
        print(f"\nFinal cookies: {len(final_cookies)}")
        print(f"Total unique books across all pages: {len(set(all_titles))}")
        print(f"Total books (with duplicates): {len(all_titles)}")

        # Demonstrate that context is still usable after closing individual pages
        verification_page = await context.new_page()
        try:
            await verification_page.goto(
                "https://books.toscrape.com/", wait_until="domcontentloaded"
            )
            cat_count = await verification_page.evaluate(
                "document.querySelectorAll('.side_categories li ul li').length"
            )
            print(
                f"\nVerification: homepage has {cat_count} categories (context still alive ✓)"
            )
        finally:
            await verification_page.close()

    finally:
        await context.close()
        print("\nContext closed. Demo complete.")


if __name__ == "__main__":
    asyncio.run(main())
