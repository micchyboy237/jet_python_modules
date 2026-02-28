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
    url: str,
    filter_chain: FilterChain | None = None,
    max_depth: int = 1,
    wait_for: str | None = None,
    wait_for_timeout: int | None = 5000,
):
    js_commands = [
        "window.scrollTo(0, document.body.scrollHeight);",
    ]
    config = CrawlerRunConfig(
        js_code=js_commands,
        wait_for=wait_for,
        wait_for_timeout=wait_for_timeout,
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
    parser.add_argument(
        "url",
        help="Root URL to crawl (required)",
    )
    parser.add_argument(
        "--max-depth",
        "-d",
        type=int,
        default=1,
        help="Maximum crawl depth (default: %(default)s)",
    )
    parser.add_argument(
        "--allowed-domains",
        "-a",
        nargs="*",
        default=None,
        help="Domains allowed for crawling",
    )
    parser.add_argument(
        "--blocked-domains",
        "-b",
        nargs="*",
        default=None,
        help="Domains to block from crawling",
    )
    parser.add_argument(
        "--url-patterns",
        "-p",
        nargs="*",
        default=None,
        help="URL patterns to allow (e.g., '*2024*')",
    )
    parser.add_argument(
        "--content-types",
        "-t",
        nargs="*",
        default=["text/html"],
        help="Allowed content types for crawling (default: %(default)s)",
    )
    parser.add_argument(
        "--wait-for",
        "-w",
        type=str,
        default=None,
        help="CSS selector or JS condition to wait for before extracting content",
    )
    parser.add_argument(
        "--wait-for-timeout",
        "-W",
        type=int,
        default=5000,
        help="Timeout (ms) for the wait_for condition (default: 5000; None = uses page_timeout)",
    )

    args = parser.parse_args()

    filters = []
    if args.url_patterns is not None:
        filters.append(URLPatternFilter(patterns=args.url_patterns))
    if args.allowed_domains is not None or args.blocked_domains is not None:
        filters.append(
            DomainFilter(
                allowed_domains=args.allowed_domains
                if args.allowed_domains is not None
                else [],
                blocked_domains=args.blocked_domains
                if args.blocked_domains is not None
                else [],
            )
        )
    if args.content_types is not None:
        filters.append(ContentTypeFilter(allowed_types=args.content_types))
    chain = FilterChain(filters)
    results_multi_filter = asyncio.run(
        crawl_with_filters(
            args.url,
            chain,
            max_depth=args.max_depth,
            wait_for=args.wait_for,
            wait_for_timeout=args.wait_for_timeout,
        )
    )
    print_results(
        results_multi_filter,
        f"{args.url_patterns} + {args.allowed_domains} + {args.content_types}",
    )

    save_file(results_multi_filter, OUTPUT_DIR / "results_multi_filter.json")
    save_file(
        [u for u in results_multi_filter], OUTPUT_DIR / "results_multi_filter_urls.json"
    )

    console.print(f"\n[green]Artifacts saved → {OUTPUT_DIR}[/]")
