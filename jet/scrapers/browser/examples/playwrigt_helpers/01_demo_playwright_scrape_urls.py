"""
Demo: Synchronous parallel URL scraping with playwright_helpers.scrape_urls()

This example demonstrates how to use the synchronous scrape_urls() function
to scrape multiple URLs in parallel. The function manages browser lifecycle,
handles retries, scrolling, screenshots, and caching automatically.

Features demonstrated:
- Parallel scraping with num_parallel parameter
- Progress bar display
- Screenshot capture
- HTML content extraction
- Scrolling strategies for infinite scroll pages
- Retry logic with exponential backoff
- Redis caching (optional)
"""

import shutil
from pathlib import Path

from jet.logger import logger
from jet.scrapers.browser.playwright_helpers import scrape_urls

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


def main():
    """Run synchronous parallel scraping demo."""
    logger.info("=" * 80)
    logger.info("Starting synchronous parallel URL scraping demo")
    logger.info("=" * 80)

    print(f"\nScraping {len(URLS_TO_SCRAPE)} URLs with parallel execution...")
    print(f"Output directory: {OUTPUT_DIR}\n")

    # Call scrape_urls with various options
    results = list(
        scrape_urls(
            urls=URLS_TO_SCRAPE,
            num_parallel=3,  # Process up to 3 URLs concurrently
            limit=None,  # No limit on number of successful scrapes
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
        )
    )

    # Process results
    success_count = 0
    failed_count = 0

    for result in results:
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


if __name__ == "__main__":
    main()
