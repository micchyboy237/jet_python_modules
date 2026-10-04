"""
Demo: Asynchronous parallel URL scraping with playwright_helpers.ascrape_urls()

This example demonstrates how to use the asynchronous ascrape_urls() function
to scrape multiple URLs in parallel using async/await. This approach is more
efficient for I/O-bound operations and allows better resource utilization.

Features demonstrated:
- Async iterator pattern for streaming results
- Parallel scraping with num_parallel parameter
- Progress bar display with tqdm_asyncio
- Screenshot capture
- HTML content extraction
- Early termination with limit parameter
- Context reuse for multiple scrape calls
- Proper cleanup of browser resources
"""

import asyncio
import shutil
from pathlib import Path

from jet.logger import logger
from jet.scrapers.browser.playwright_helpers import (
    ascrape_urls,
    setup_async_browser_session,
)

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Example URLs to scrape (mix of static and dynamic content)
URLS_TO_SCRAPE = [
    "https://web-scraping.dev/testimonials",
    "https://quotes.toscrape.com/scroll",
    "https://webscraper.io/test-sites/scroll",
    "https://the-internet.herokuapp.com/infinite_scroll",
    "https://scrapethissite.com/pages/ajax",
]


async def scrape_with_context_reuse():
    """
    Demo: Reuse a single browser context across multiple scrape calls.

    This is more efficient than creating a new browser for each call,
    especially when scraping many URLs or making multiple scrape calls.
    """
    logger.info("=" * 80)
    logger.info("Async scraping with context reuse demo")
    logger.info("=" * 80)

    # Create a persistent browser context
    logger.info("Creating persistent browser context...")
    context = await setup_async_browser_session(headless=True)

    try:
        print(f"\nScraping {len(URLS_TO_SCRAPE)} URLs with context reuse...")
        print(f"Output directory: {OUTPUT_DIR}\n")

        success_count = 0
        failed_count = 0

        # Use ascrape_urls with the reused context
        async for result in ascrape_urls(
            urls=URLS_TO_SCRAPE,
            num_parallel=3,  # Process up to 3 URLs concurrently
            limit=None,  # No limit on successful scrapes
            show_progress=True,  # Show tqdm progress bar
            timeout=10000,  # 10 second timeout per page load
            max_retries=2,  # Retry failed attempts up to 2 times
            with_screenshot=True,  # Capture full-page screenshots
            headless=True,  # Run browser in headless mode
            wait_for_js=True,  # Wait for JavaScript to render
            use_cache=False,  # Don't use Redis cache for this demo
            scroll_strategy="until_stable",  # Scroll until page height stabilizes
            scroll_mode="increment",  # Scroll by viewport increments
            scroll_max_attempts=15,  # Maximum scroll attempts
            scroll_delay_ms=1400,  # Delay between scrolls in milliseconds
            context=context,  # Reuse the existing context
        ):
            url = result["url"]
            status = result["status"]

            if status == "completed":
                success_count += 1

                # Save HTML content
                safe_filename = (
                    url.replace("https://", "")
                    .replace("http://", "")
                    .replace("/", "_")
                    .replace(".", "_")
                )
                html_path = OUTPUT_DIR / f"{safe_filename}.html"

                if result["html"]:
                    with open(html_path, "w", encoding="utf-8") as f:
                        f.write(result["html"])
                    logger.success(f"✓ Saved HTML: {html_path.name}")

                # Save screenshot if available
                if result["screenshot"]:
                    screenshot_path = OUTPUT_DIR / f"{safe_filename}.png"
                    with open(screenshot_path, "wb") as f:
                        f.write(result["screenshot"])
                    logger.success(f"✓ Saved screenshot: {screenshot_path.name}")

            elif status in ["failed_no_html", "failed_error"]:
                failed_count += 1
                logger.error(f"✗ Failed to scrape: {url} (status: {status})")

        # Summary
        print("\n" + "=" * 80)
        logger.info("Scraping Complete!")
        logger.info(f"Total URLs: {len(URLS_TO_SCRAPE)}")
        logger.success(f"Successful: {success_count}")
        if failed_count > 0:
            logger.warning(f"Failed: {failed_count}")
        logger.info(f"Output saved to: {OUTPUT_DIR}")
        logger.info("=" * 80)

    finally:
        # Clean up the context
        logger.info("Closing browser context...")
        await context.close()


async def scrape_with_limit():
    """
    Demo: Stop after N successful scrapes using the limit parameter.

    Useful when you only need a subset of pages or want to test with
    a small number of URLs first.
    """
    logger.info("=" * 80)
    logger.info("Async scraping with early termination demo")
    logger.info("=" * 80)

    print(f"\nScraping with limit=2 (will stop after 2 successful scrapes)...")
    print(f"Output directory: {OUTPUT_DIR}\n")

    success_count = 0

    async for result in ascrape_urls(
        urls=URLS_TO_SCRAPE,
        num_parallel=2,
        limit=2,  # Stop after 2 successful scrapes
        show_progress=True,
        timeout=10000,
        max_retries=1,
        with_screenshot=False,  # Skip screenshots for faster execution
        headless=True,
        wait_for_js=True,
        use_cache=False,
        scroll_strategy="none",  # Skip scrolling for faster execution
    ):
        if result["status"] == "completed":
            success_count += 1
            logger.success(f"✓ Completed: {result['url']}")

            # Just save HTML, no screenshot
            safe_filename = (
                result["url"]
                .replace("https://", "")
                .replace("http://", "")
                .replace("/", "_")
                .replace(".", "_")
            )
            html_path = OUTPUT_DIR / f"{safe_filename}_limited.html"

            if result["html"]:
                with open(html_path, "w", encoding="utf-8") as f:
                    f.write(result["html"])

    print(f"\nStopped after {success_count} successful scrapes (limit was 2)")
    logger.info("=" * 80)


async def main():
    """Run all async scraping demos."""
    # Demo 1: Context reuse
    await scrape_with_context_reuse()

    print("\n\n")

    # Demo 2: Early termination with limit
    await scrape_with_limit()


if __name__ == "__main__":
    asyncio.run(main())
