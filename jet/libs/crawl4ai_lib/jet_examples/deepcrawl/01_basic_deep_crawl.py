import argparse
import asyncio
import shutil
import time
from pathlib import Path

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from jet.file.utils import save_file

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)


# 1️⃣ Basic Deep Crawl Setup
async def basic_deep_crawl(url: str, max_depth: int = 2):
    """
    PART 1: Basic Deep Crawl setup - Demonstrates a simple two-level deep crawl.

    This function shows:
    - How to set up BFSDeepCrawlStrategy (Breadth-First Search)
    - Setting depth and domain parameters
    - Processing the results to show the hierarchy
    """
    print("\n===== BASIC DEEP CRAWL SETUP =====")

    # Configure a 2-level deep crawl using Breadth-First Search strategy
    # max_depth=2 means: initial page (depth 0) + 2 more levels
    # include_external=False means: only follow links within the same domain
    config = CrawlerRunConfig(
        deep_crawl_strategy=BFSDeepCrawlStrategy(
            max_depth=max_depth, include_external=False
        ),
        scraping_strategy=LXMLWebScrapingStrategy(),
        verbose=True,  # Show progress during crawling
    )

    async with AsyncWebCrawler() as crawler:
        results = await crawler.arun(url=url, config=config)
        return results


def group_pages_by_depth(results):
    # Group results by depth to visualize the crawl tree
    pages_by_depth = {}
    for result in results:
        depth = result.metadata.get("depth", 0)
        if depth not in pages_by_depth:
            pages_by_depth[depth] = []
        pages_by_depth[depth].append(result.url)
    return pages_by_depth


# Execute the tutorial when run directly
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a basic deep crawl using the Crawl4AI library."
    )
    parser.add_argument(
        "url",
        nargs="?",
        default="https://docs.crawl4ai.com",
        help="The starting URL for the crawl (default: https://docs.crawl4ai.com)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=2,
        help="Maximum crawl depth (default: 2)",
    )

    args = parser.parse_args()

    start_time = time.perf_counter()

    results = asyncio.run(basic_deep_crawl(args.url, args.max_depth))

    print(f"✅ Crawled {len(results)} pages total")

    pages_by_depth = group_pages_by_depth(results)

    # Display crawl structure by depth
    for depth, urls in sorted(pages_by_depth.items()):
        print(f"\nDepth {depth}: {len(urls)} pages")
        # Show first 3 URLs for each depth as examples
        for url in urls[:3]:
            print(f"  → {url}")
        if len(urls) > 3:
            print(f"  ... and {len(urls) - 3} more")

    print(
        f"\n✅ Performance: {len(results)} pages in {time.perf_counter() - start_time:.2f} seconds"
    )

    save_file(pages_by_depth, OUTPUT_DIR / "pages_by_depth.json")
    save_file(results, OUTPUT_DIR / "results.json")
