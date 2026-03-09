import argparse
import asyncio
import shutil
from pathlib import Path

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from crawl4ai.deep_crawling import (
    BestFirstCrawlingStrategy,
    BFSDeepCrawlStrategy,
    DFSDeepCrawlStrategy,
)
from crawl4ai.deep_crawling.filters import (
    ContentRelevanceFilter,  # ← added (query-based relevance)
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
    strategy_type: str = "bfs",
    max_depth: int = 2,
    wait_for: str | None = None,
    wait_for_timeout: int | None = 5000,
    relevance_query: str | None = None,
    relevance_threshold: float = 0.5,
):
    js_commands = [
        "window.scrollTo(0, document.body.scrollHeight);",
    ]

    # Build filter chain (including optional relevance filter)
    filters = filter_chain.filters if filter_chain else []
    if relevance_query:
        filters.append(
            ContentRelevanceFilter(
                query=relevance_query,
                threshold=relevance_threshold,
            )
        )
    final_chain = FilterChain(filters) if filters else None

    # Choose strategy
    if strategy_type == "bestfirst":
        strategy = BestFirstCrawlingStrategy(
            max_depth=max_depth,
            include_external=False,
            filter_chain=final_chain,
            max_pages=100,  # safety limit — adjust as needed
        )
    elif strategy_type == "dfs":
        strategy = DFSDeepCrawlStrategy(
            max_depth=max_depth,
            include_external=False,
            filter_chain=final_chain,
        )
    else:  # default bfs
        strategy = BFSDeepCrawlStrategy(
            max_depth=max_depth,
            include_external=False,
            filter_chain=final_chain,
        )

    config = CrawlerRunConfig(
        js_code=js_commands,
        wait_for=wait_for,
        wait_for_timeout=wait_for_timeout,
        deep_crawl_strategy=strategy,
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
    parser = argparse.ArgumentParser(
        description="Crawl4AI deep crawler with filters + relevance query support"
    )
    parser.add_argument("url", help="Root URL to crawl (required)")
    parser.add_argument(
        "--max-depth",
        "-d",
        type=int,
        default=2,
        help="Maximum crawl depth (default: %(default)s)",
    )
    parser.add_argument(
        "--strategy",
        "-s",
        choices=["bfs", "dfs", "bestfirst"],
        default="bfs",
        help="Crawl strategy: bfs (default), dfs, or bestfirst (recommended for relevance)",
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
        help="URL patterns to allow (e.g., '*2024*' '*blog*')",
    )
    parser.add_argument(
        "--content-types",
        "-t",
        nargs="*",
        default=["text/html"],
        help="Allowed content types (default: %(default)s)",
    )
    parser.add_argument(
        "--relevance-query",
        "-q",
        type=str,
        default=None,
        help="Semantic relevance query — pages must match this topic (uses ContentRelevanceFilter)",
    )
    parser.add_argument(
        "--relevance-threshold",
        type=float,
        default=0.5,
        help="Minimum relevance score (0.0–1.0) when using --relevance-query (default: 0.5)",
    )
    parser.add_argument(
        "--wait-for",
        "-w",
        type=str,
        default=None,
        help="CSS selector or JS condition to wait for",
    )
    parser.add_argument(
        "--wait-for-timeout",
        "-W",
        type=int,
        default=5000,
        help="Timeout (ms) for wait_for (default: 5000)",
    )

    args = parser.parse_args()

    # Build base filter chain
    filters = []
    if args.url_patterns:
        filters.append(URLPatternFilter(patterns=args.url_patterns))
    if args.allowed_domains or args.blocked_domains:
        filters.append(
            DomainFilter(
                allowed_domains=args.allowed_domains or [],
                blocked_domains=args.blocked_domains or [],
            )
        )
    if args.content_types:
        filters.append(ContentTypeFilter(allowed_types=args.content_types))

    chain = FilterChain(filters) if filters else None

    # Run crawl (relevance filter is added inside crawl_with_filters if query given)
    results = asyncio.run(
        crawl_with_filters(
            url=args.url,
            filter_chain=chain,
            strategy_type=args.strategy,
            max_depth=args.max_depth,
            wait_for=args.wait_for,
            wait_for_timeout=args.wait_for_timeout,
            relevance_query=args.relevance_query,
            relevance_threshold=args.relevance_threshold,
        )
    )

    # Print summary
    filter_desc = []
    if args.url_patterns:
        filter_desc.append(f"patterns={args.url_patterns}")
    if args.allowed_domains or args.blocked_domains:
        filter_desc.append(f"domains={args.allowed_domains}/{args.blocked_domains}")
    if args.content_types != ["text/html"]:
        filter_desc.append(f"types={args.content_types}")
    if args.relevance_query:
        filter_desc.append(
            f"relevance='{args.relevance_query}' (≥{args.relevance_threshold})"
        )
    if not filter_desc:
        filter_desc = ["no filters"]

    title = f"Deep crawl ({args.strategy.upper()}) — {' + '.join(filter_desc)}"
    print_results(results, title)

    # Save artifacts
    save_file(results, OUTPUT_DIR / "results.json")
    save_file([r.url for r in results], OUTPUT_DIR / "urls.json")

    console.print(f"\n[green]Artifacts saved → {OUTPUT_DIR}[/]")
