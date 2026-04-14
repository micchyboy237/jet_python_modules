"""
Streaming Crawl with Filters

Usage Examples
--------------

python streaming_crawl_with_llm_query.py "https://missav.ws/en" -u "*.ws/en/*-*" -s 4
"""

import argparse
import asyncio
import random
import shutil
import sys
import time
from math import inf as infinity
from pathlib import Path

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, LLMExtractionStrategy
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.deep_crawling.filters import (
    ContentTypeFilter,
    DomainFilter,
    FilterChain,
    URLPatternFilter,
)
from jet.file.utils import save_file
from jet.libs.crawl4ai_lib.adaptive_config import get_llm_config
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
    max_depth: int = 2,
    wait_for: str | None = None,
    wait_for_timeout: int | None = 5000,
    max_pages: int = infinity,
    polite_delay_range: tuple[float, float] = (2.5, 6.0),  # seconds
    query: str | None = None,
):
    js_commands = [
        "window.scrollTo(0, document.body.scrollHeight);",
    ]

    llm_strategy = None
    if query:
        llm_strategy = LLMExtractionStrategy(
            llm_config=get_llm_config(strategy="llm"),
            instruction=query,
        )

    config = CrawlerRunConfig(
        js_code=js_commands,
        wait_for=wait_for,
        wait_for_timeout=wait_for_timeout or 30000,  # more generous default
        page_timeout=60000,  # 60s – missav is slow
        deep_crawl_strategy=BFSDeepCrawlStrategy(
            max_depth=max_depth,
            include_external=False,
            filter_chain=filter_chain,
            max_pages=max_pages,
        ),
        scraping_strategy=LXMLWebScrapingStrategy(),
        cache_mode="BYPASS",
        verbose=True,
        stream=True,
        extraction_strategy=llm_strategy,
        capture_network_requests=True,
        capture_console_messages=True,
        log_console=True,
    )

    async with AsyncWebCrawler() as crawler:
        start = time.perf_counter()
        results = []
        first_time = None

        # This assumes 'crawl_results' is async generator
        idx = 0
        async for result in await crawler.arun(url=url, config=config):
            results.append(result)
            idx += 1
            if idx == 1:
                first_time = time.perf_counter() - start

            console.print(f" → #{idx:3d} | count: {len(results):3d} | {result.url}")
            sys.stdout.flush()  # <-- This is the key for "flush each streamed chunk"

            # Optional: show captured logs from this page
            if result.console_messages:
                for msg in result.console_messages[:5]:  # limit to avoid spam
                    console.print(f"   [dim]console: {msg}[/dim]")

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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "url",
        help="Root URL to crawl (required)",
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Optional query string for LLM extraction (e.g., a question, or extraction instruction).",
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
        help="Maximum number of path segments allowed in URLs (default: None)",
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
    if args.max_segments is not None:
        filters.append(MaxPathSegmentsFilter(max_segments=args.max_segments))
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
    results, first_time, total_time = asyncio.run(
        streaming_crawl_with_filters(
            args.url,
            chain,
            max_depth=args.max_depth,
            max_pages=args.max_pages,
            wait_for=args.wait_for,
            wait_for_timeout=args.wait_for_timeout,
            query=args.query,
        )
    )
    summary_data = {
        "mode": "streaming",
        "pages": len(results),
        "time_to_first": first_time,
        "total_time": total_time,
    }

    print_summary("Streaming", len(results), first_time, total_time)

    save_file(results, OUTPUT_DIR / "results.json")
    save_file([u for u in results], OUTPUT_DIR / "result_links.json")
    save_file(
        summary_data,
        OUTPUT_DIR / "counts.json",
    )

    console.print(f"\n[green]Artifacts saved → {OUTPUT_DIR}[/]")
