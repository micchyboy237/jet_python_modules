import argparse
import asyncio
import shutil
from pathlib import Path

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.deep_crawling.filters import (
    ContentTypeFilter,
    DomainFilter,
    FilterChain,
    URLPatternFilter,
)
from jet.file.utils import save_file
from rich.console import Console

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

console = Console(highlight=True)


async def crawl_with_filters(
    url: str, filter_chain: FilterChain | None = None, max_depth: int = 1
):
    config = CrawlerRunConfig(
        deep_crawl_strategy=BFSDeepCrawlStrategy(
            max_depth=max_depth,
            include_external=False,
            filter_chain=filter_chain,
        ),
        scraping_strategy=LXMLWebScrapingStrategy(),
        cache_mode="BYPASS",
        verbose=True,
    )
    async with AsyncWebCrawler() as crawler:
        return await crawler.arun(url=url, config=config)


def print_results(results, title: str):
    console.rule(title)
    console.print(f"[bold green]Crawled {len(results)} pages[/]")
    for r in results[:6]:
        console.print(f"  • {r.url}")
    if len(results) > 6:
        console.print(f"  … and {len(results) - 6} more")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("url", nargs="?", default="https://docs.crawl4ai.com")
    parser.add_argument(
        "--max-depth",
        "-d",
        type=int,
        default=3,
        help="Maximum crawl depth (default: 3)",
    )
    args = parser.parse_args()

    # Example 1: URL pattern only
    chain1 = FilterChain([URLPatternFilter(patterns=["*core*"])])
    results_url_pattern_only = asyncio.run(
        crawl_with_filters(args.url, chain1, max_depth=args.max_depth)
    )
    print_results(results_url_pattern_only, "Only URLs containing 'core'")

    save_file(results_url_pattern_only, OUTPUT_DIR / "results_url_pattern_only.json")
    save_file(
        [u for u in results_url_pattern_only],
        OUTPUT_DIR / "results_url_pattern_only_urls.json",
    )

    # Example 2: Multi-filter chain (different site to show domain filter)
    chain2 = FilterChain(
        [
            URLPatternFilter(patterns=["*2024*"]),
            DomainFilter(
                allowed_domains=["techcrunch.com"],
                blocked_domains=["guce.techcrunch.com", "oidc.techcrunch.com"],
            ),
            ContentTypeFilter(allowed_types=["text/html"]),
        ]
    )
    results_multi_filter = asyncio.run(
        crawl_with_filters("https://techcrunch.com", chain2, max_depth=args.max_depth)
    )
    print_results(results_multi_filter, "2024 + techcrunch.com + html")

    save_file(results_multi_filter, OUTPUT_DIR / "results_multi_filter.json")
    save_file(
        [u for u in results_multi_filter], OUTPUT_DIR / "results_multi_filter_urls.json"
    )

    console.print(f"\n[green]Artifacts saved → {OUTPUT_DIR}[/]")
