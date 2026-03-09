import argparse
import asyncio
import shutil
from pathlib import Path

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from crawl4ai.deep_crawling import (
    BestFirstCrawlingStrategy,  # ← Recommended for relevance
)
from crawl4ai.deep_crawling.filters import (
    ContentRelevanceFilter,  # ← NEW: BM25 content relevance
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
from rich.console import Console

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

console = Console(highlight=True)


async def crawl_with_relevance(
    url: str,
    filter_chain: FilterChain | None = None,
    url_scorer=None,
    max_depth: int = 2,
    max_pages: int = 50,  # Safety limit
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
        deep_crawl_strategy=BestFirstCrawlingStrategy(
            max_depth=max_depth,
            max_pages=max_pages,
            include_external=False,
            filter_chain=filter_chain,
            url_scorer=url_scorer,  # Keyword / composite scorer for prioritization
        ),
        # If you prefer BFS but with scoring (less intelligent order):
        # deep_crawl_strategy=BFSDeepCrawlStrategy(
        #     max_depth=max_depth,
        #     include_external=False,
        #     filter_chain=filter_chain,
        #     url_scorer=url_scorer,
        #     score_threshold=0.3,          # Optional hard cutoff
        # ),
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
        score_str = (
            f" (score: {r.metadata.get('score', 'n/a'):.2f})"
            if "metadata" in dir(r)
            else ""
        )
        console.print(f" • {r.url}{score_str}")
    if len(results) > 6:
        console.print(f" … and {len(results) - 6} more")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("url", help="Root URL to crawl (required)")
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Query string for keyword relevance scoring & optional content filter",
    )
    parser.add_argument(
        "--max-depth",
        "-d",
        type=int,
        default=2,
        help="Maximum crawl depth (default: %(default)s)",
    )
    parser.add_argument(
        "--max-pages",
        "-m",
        type=int,
        default=50,
        help="Maximum total pages to crawl (safety limit)",
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
    args = parser.parse_args()

    # Hard filters (reject before fetching)
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

    # Optional: Add content-level relevance filter (BM25 on fetched content)
    if args.query:
        filters.append(
            ContentRelevanceFilter(
                query=args.query,
                threshold=0.45,  # Adjust: higher = stricter
            )
        )

    chain = FilterChain(filters) if filters else None

    # Relevance scorer for prioritization (using available scorers)
    url_scorer = None
    if args.query:
        keywords = args.query.lower().split()  # Simple split; improve as needed
        keyword_scorer = KeywordRelevanceScorer(keywords=keywords, weight=0.8)

        # Optional: Combine with other scorers for better prioritization
        scorers = [
            keyword_scorer,
            PathDepthScorer(optimal_depth=3, weight=0.4),  # Prefer medium-depth pages
            FreshnessScorer(weight=0.3, current_year=2026),  # Prefer recent URLs
        ]
        url_scorer = CompositeScorer(scorers=scorers, normalize=True)

        console.print(f"[cyan]Using relevance strategy with query: {args.query}[/]")
        console.print(f"   → Keywords: {keywords}")
    else:
        console.print("[yellow]No query → crawling without relevance scoring[/]")

    results = asyncio.run(
        crawl_with_relevance(
            args.url,
            filter_chain=chain,
            url_scorer=url_scorer,
            max_depth=args.max_depth,
            max_pages=args.max_pages,
            wait_for=args.wait_for,
            wait_for_timeout=args.wait_for_timeout,
        )
    )

    title = f"Relevance crawl ({args.query or 'no query'}) + patterns={args.url_patterns} + domains={args.allowed_domains}"
    print_results(results, title)

    save_file(results, OUTPUT_DIR / "results.json")
    save_file([r.url for r in results], OUTPUT_DIR / "urls.json")
    console.print(f"\n[green]Artifacts saved → {OUTPUT_DIR}[/]")
