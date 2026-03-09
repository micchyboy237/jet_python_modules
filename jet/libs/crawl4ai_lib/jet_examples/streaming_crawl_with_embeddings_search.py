"""
Streaming Crawl with Filters & Optional Relevance Scoring

Usage Examples
--------------

# Basic pattern-based crawl
python streaming_crawl_with_embeddings_search.py "https://missav.ws/en" -u "*.ws/en/*-*" -s 4 --query "mother son incest"

# Relevance-guided (best-first) with keywords
python streaming_crawl_with_embeddings_search.py "https://example.com" --query "tutorial guide async" --max-depth 3 --max-pages 40
"""

import argparse
import asyncio
import random
import shutil
import time
from math import inf as infinity
from pathlib import Path

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from crawl4ai.deep_crawling import BestFirstCrawlingStrategy, BFSDeepCrawlStrategy
from crawl4ai.deep_crawling.filters import (
    ContentRelevanceFilter,
    ContentTypeFilter,
    DomainFilter,
    FilterChain,
    URLPatternFilter,
)
from crawl4ai.deep_crawling.scorers import (
    CompositeScorer,
    FreshnessScorer,
    KeywordRelevanceScorer,
    PathDepthScorer,
)
from jet.file.utils import save_file
from jet.libs.crawl4ai_lib.custom_filters import MaxPathSegmentsFilter
from rich.console import Console
from rich.table import Table

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

console = Console(highlight=True)


async def streaming_crawl_with_filters(
    url: str,
    filter_chain: FilterChain | None = None,
    url_scorer=None,
    max_depth: int = 2,
    max_pages: int = infinity,
    wait_for: str | None = None,
    wait_for_timeout: int | None = 5000,
    polite_delay_range: tuple[float, float] = (2.5, 6.0),  # seconds
    use_best_first: bool = True,
):
    js_commands = [
        "window.scrollTo(0, document.body.scrollHeight);",
    ]

    strategy_cls = (
        BestFirstCrawlingStrategy
        if use_best_first and url_scorer
        else BFSDeepCrawlStrategy
    )

    strategy_kwargs = {
        "max_depth": max_depth,
        "include_external": False,
        "filter_chain": filter_chain,
        "max_pages": max_pages if max_pages != infinity else None,
    }

    if use_best_first and url_scorer:
        strategy_kwargs["url_scorer"] = url_scorer
    # BFS supports score_threshold; BestFirst prioritizes instead
    elif not use_best_first:
        strategy_kwargs["score_threshold"] = 0.25  # optional hard cutoff

    config = CrawlerRunConfig(
        js_code=js_commands,
        wait_for=wait_for,
        wait_for_timeout=wait_for_timeout or 30000,
        page_timeout=60000,  # generous for slow sites
        deep_crawl_strategy=strategy_cls(**strategy_kwargs),
        scraping_strategy=LXMLWebScrapingStrategy(),
        cache_mode="BYPASS",
        verbose=True,
        stream=True,
    )

    async with AsyncWebCrawler() as crawler:
        start = time.perf_counter()
        results = []
        first_time = None
        idx = 0

        async for result in await crawler.arun(url=url, config=config):
            results.append(result)
            idx += 1
            if idx == 1:
                first_time = time.perf_counter() - start

            score_str = (
                f" | score: {result.metadata.get('score', 'n/a'):.2f}"
                if "metadata" in dir(result)
                else ""
            )
            console.print(f"  → #{idx:3d} | {result.url}{score_str}")

            if polite_delay_range:
                delay = random.uniform(*polite_delay_range)
                await asyncio.sleep(delay)

        total_time = time.perf_counter() - start
        return results, first_time, total_time


def print_summary(mode: str, count: int, first_time: float | None, total_time: float):
    table = Table(title=f"{mode} Mode Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_row("Pages crawled", f"{count}")
    if first_time is not None:
        table.add_row("Time to first result", f"{first_time:.2f} s")
    table.add_row("Total time", f"{total_time:.2f} s")
    console.print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Streaming deep crawl with filters & optional keyword relevance."
    )
    parser.add_argument("url", help="Root URL to crawl (required)")
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Query/keywords for relevance scoring (enables BestFirstCrawlingStrategy)",
    )
    parser.add_argument(
        "--max-depth",
        "-d",
        type=int,
        default=2,
        help="Maximum crawl depth (default: %(default)s)",
    )
    parser.add_argument(
        "--max-segments",
        "-s",
        type=int,
        default=None,
        help="Maximum number of path segments allowed in URLs",
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
        "-u",
        nargs="*",
        default=None,
        help="URL patterns to allow (e.g., '*2024*')",
    )
    parser.add_argument(
        "--max-pages",
        "-p",
        type=int,
        default=20,
        help="Maximum number of pages to crawl (default: %(default)s)",
    )
    parser.add_argument(
        "--content-types",
        "-t",
        nargs="*",
        default=["text/html"],
        help="Allowed content types (default: %(default)s)",
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
    parser.add_argument(
        "--no-best-first",
        action="store_true",
        help="Force BFS instead of BestFirst (even with --query)",
    )

    args = parser.parse_args()

    # Hard filters (pre-fetch rejection)
    filters = []
    if args.max_segments is not None:
        filters.append(MaxPathSegmentsFilter(max_segments=args.max_segments))
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

    # Optional content-level filter (post-fetch BM25 relevance)
    if args.query:
        filters.append(
            ContentRelevanceFilter(
                query=args.query,
                threshold=0.4,  # adjust as needed (higher = stricter)
            )
        )

    chain = FilterChain(filters) if filters else None

    # Relevance scorer (for BestFirst prioritization)
    url_scorer = None
    if args.query and not args.no_best_first:
        keywords = args.query.lower().split()
        keyword_scorer = KeywordRelevanceScorer(keywords=keywords, weight=0.75)

        # Optional composite for better prioritization
        scorers = [
            keyword_scorer,
            PathDepthScorer(optimal_depth=3, weight=0.35),
            FreshnessScorer(weight=0.25, current_year=2026),
        ]
        url_scorer = CompositeScorer(scorers=scorers, normalize=True)

        console.print("[cyan]Using BestFirst strategy with relevance scoring[/]")
        console.print(f"   Keywords: {keywords}")
    else:
        console.print("[yellow]No query or --no-best-first → using BFS[/]")

    results, first_time, total_time = asyncio.run(
        streaming_crawl_with_filters(
            args.url,
            filter_chain=chain,
            url_scorer=url_scorer,
            max_depth=args.max_depth,
            max_pages=args.max_pages,
            wait_for=args.wait_for,
            wait_for_timeout=args.wait_for_timeout,
            use_best_first=not args.no_best_first,
        )
    )

    mode = "Best-First Relevance" if url_scorer else "Streaming BFS"
    print_summary(mode, len(results), first_time, total_time)

    summary_data = {
        "mode": mode,
        "pages": len(results),
        "time_to_first": first_time,
        "total_time": total_time,
        "query": args.query,
    }

    save_file(results, OUTPUT_DIR / "results.json")
    save_file([r.url for r in results], OUTPUT_DIR / "result_links.json")
    save_file(summary_data, OUTPUT_DIR / "summary.json")

    console.print(f"\n[green]Artifacts saved → {OUTPUT_DIR}[/]")
