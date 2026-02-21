import argparse
import asyncio
import shutil
from pathlib import Path
from typing import List

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.deep_crawling.filters import (
    ContentRelevanceFilter,
    FilterChain,
    SEOFilter,
)
from jet.file.utils import save_file
from rich.console import Console

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

console = Console()


async def crawl_with_advanced_filters(url: str, filter_chain: FilterChain):
    config = CrawlerRunConfig(
        deep_crawl_strategy=BFSDeepCrawlStrategy(
            max_depth=1,
            include_external=False,
            filter_chain=filter_chain,
        ),
        scraping_strategy=LXMLWebScrapingStrategy(),
        verbose=True,
        cache_mode="BYPASS",
    )
    async with AsyncWebCrawler() as crawler:
        return await crawler.arun(url=url, config=config)


def parse_keywords(value: str | None, fallback: List[str]) -> List[str]:
    """Parse comma-separated keywords from CLI argument."""
    if not value:
        return fallback
    cleaned = [k.strip() for k in value.split(",") if k.strip()]
    return cleaned if cleaned else fallback


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demonstrate advanced filters (SEO + semantic relevance)"
    )
    parser.add_argument("url", nargs="?", default="https://docs.crawl4ai.com")
    parser.add_argument(
        "--seo-keywords",
        "-k",
        type=str,
        default=None,
        help="Comma-separated keywords for SEO filter (head section match)",
    )
    parser.add_argument(
        "--query", "-q", type=str, default=None, help="Semantic relevance query"
    )
    args = parser.parse_args()

    # SEO filter example
    seo_keywords = parse_keywords(
        args.seo_keywords, ["crawl", "dynamic", "async", "javascript", "deep"]
    )
    seo_filter = SEOFilter(threshold=0.5, keywords=seo_keywords)

    console.rule("SEO Filter Configuration")
    console.print(
        f"Keywords: [yellow]{', '.join(seo_keywords)}[/yellow] (threshold 0.5)"
    )

    r_seo = asyncio.run(
        crawl_with_advanced_filters(args.url, FilterChain([seo_filter]))
    )
    console.rule("SEO Filter (keywords in head)")
    console.print(f"Found {len(r_seo)} pages")
    for r in r_seo[:5]:
        console.print(f"  • {r.url}")

    # Content relevance (semantic)
    query = (
        args.query or "How to configure and use deep crawling strategies in Crawl4AI"
    )
    relevance_filter = ContentRelevanceFilter(query=query, threshold=0.65)

    console.rule("Semantic Relevance Filter Configuration")
    console.print(f"Query: [yellow]{query}[/yellow] (threshold 0.65)")

    r_rel = asyncio.run(
        crawl_with_advanced_filters(args.url, FilterChain([relevance_filter]))
    )
    console.rule("Semantic Relevance Filter")
    console.print(f"Found {len(r_rel)} relevant pages")
    for r in r_rel:
        score = r.metadata.get("relevance_score", 0)
        console.print(f"  • {score:.2f} | {r.url}")

    save_file([r.url for r in r_rel], OUTPUT_DIR / "relevant_pages.txt")
